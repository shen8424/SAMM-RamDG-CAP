from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_

class RamDG(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True,
                 is_train = True
                 ):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        text_encoder = './pytorch_model.bin'
        self.text_encoder = BertForTokenClassification.from_pretrained(text_encoder, config=bert_config, label_smoothing=config['label_smoothing'])      
        
        
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         
        self.extra_vision_proj = nn.Linear(vision_width, embed_dim)
        self.extra_text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.extra_temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=3)

        # momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.extra_vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.extra_text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.extra_model_pairs = [[self.extra_vision_proj,self.extra_vision_proj_m],
                                  [self.vision_proj,self.vision_proj_m],
                                  [self.extra_text_proj,self.extra_text_proj_m],
                                  [self.text_encoder,self.text_encoder_m],                          
                           ]
        
        self.copy_params(self.model_pairs)
        self.copy_params(self.extra_model_pairs)

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("extra_image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("extra_text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        self.register_buffer("extra_queue_ptr", torch.zeros(1, dtype=torch.long))            

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.extraimgs_fusion = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.binary_dec = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.patch_dec = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.norm_layer_2_dec = nn.LayerNorm(text_width)
        self.norm_layer_patch_dec = nn.LayerNorm(text_width)
        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.patch_2dec = self.build_2dec(text_width)
        self.patch_2cls_head = nn.Linear(text_width, 2)
        self.text_binary_proj = nn.Linear(text_width, text_width)
        self.image_binary_proj = nn.Linear(text_width, text_width)
        self.patch_proj = nn.Linear(text_width, text_width)
        self.text_decoder_fusion = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.norm_layer_decoder_fusion = nn.LayerNorm(text_width)

        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)
        if is_train:
            self.proj_model_pairs = [[self.vision_proj,self.extra_vision_proj],
                            [self.text_proj,self.extra_text_proj]
                           ]
            self.proj_model_pairs_m = [[self.vision_proj_m,self.extra_vision_proj_m],
                            [self.text_proj_m,self.extra_text_proj_m]
                           ]
            self.init_extra_proj(self.proj_model_pairs)
            self.copy_params(self.proj_model_pairs_m)
            self.extra_image_queue = self.image_queue.clone()
            self.extra_text_queue = self.text_queue.clone() 
            
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def build_2dec(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text, text_orig, fake_image_box, fake_text_pos, extra_img, image_indices, if_source_name_img, extra_text, text_indices, if_source_name_text, patch_label = None, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)
                self.extra_temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            multicls_label, real_label_pos = get_multi_label(label, image)
            loss_cncl = torch.tensor(0.0, device=image.device)
            if len(image_indices) > 0 and len(text_indices)>0 :
                extra_image_embeds = self.visual_encoder(extra_img)

                extra_text_output = self.text_encoder.bert(extra_text.input_ids, attention_mask = extra_text.attention_mask, return_dict = True, mode = 'text')            
                extra_text_embeds = extra_text_output.last_hidden_state
                
                ##############====== Loss_CNCL ======################
                extra_image_feat = F.normalize(self.extra_vision_proj(extra_image_embeds[:,0,:]),dim=-1)
                extra_text_feat = F.normalize(self.extra_text_proj(extra_text_embeds[:,0,:]),dim=-1)
                with torch.no_grad():
                    self._momentum_update(self.extra_model_pairs)
                    image_embeds_m = self.visual_encoder_m(image) 
                    image_feat_m_extra = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
                    image_feat_all = torch.cat([image_feat_m_extra.t(),self.extra_image_queue.clone().detach()],dim=1)
                    extra_image_embeds_m = self.visual_encoder_m(extra_img)
                    extra_image_feat_m = F.normalize(self.extra_vision_proj_m(extra_image_embeds_m[:,0,:]),dim=-1)

                    text_output_m = self.text_encoder_m.bert(text_orig.input_ids, attention_mask = text_orig.attention_mask,                      
                                                        return_dict = True, mode = 'text')    
                    text_feat_m_extra = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
                    text_feat_all = torch.cat([text_feat_m_extra.t(),self.extra_text_queue.clone().detach()],dim=1)
                    extra_text_output_m = self.text_encoder_m.bert(extra_text.input_ids, attention_mask = extra_text.attention_mask,                      
                                                        return_dict = True, mode = 'text')    
                    extra_text_feat_m = F.normalize(self.extra_text_proj_m(extra_text_output_m.last_hidden_state[:,0,:]),dim=-1)

                    extra_sim_i2t_m = extra_image_feat_m @ text_feat_all / self.extra_temp 
                    extra_sim_t2i_m = extra_text_feat_m @ image_feat_all / self.extra_temp

                    extra_sim_i2i_m = extra_image_feat_m @ image_feat_all / self.extra_temp 
                    extra_sim_t2t_m = extra_text_feat_m @ text_feat_all / self.extra_temp

                    extra_img_targets = torch.zeros(extra_sim_i2t_m.size())                
                    extra_img_targets = get_label(extra_img_targets, image_indices, if_source_name_img).to(image.device)

                    extra_text_targets = torch.zeros(extra_sim_t2i_m.size())                
                    extra_text_targets = get_label(extra_text_targets, text_indices, if_source_name_text).to(image.device)

                    extra_i2t_targets = alpha * F.softmax(extra_sim_i2t_m, dim=1) + (1 - alpha) * extra_img_targets
                    extra_t2i_targets = alpha * F.softmax(extra_sim_t2i_m, dim=1) + (1 - alpha) * extra_text_targets 
                    extra_i2i_targets = alpha * F.softmax(extra_sim_i2i_m, dim=1) + (1 - alpha) * extra_img_targets
                    extra_t2t_targets = alpha * F.softmax(extra_sim_t2t_m, dim=1) + (1 - alpha) * extra_text_targets

                extra_sim_i2t = extra_image_feat @ text_feat_all / self.extra_temp 
                extra_sim_t2i = extra_text_feat @ image_feat_all / self.extra_temp
                extra_sim_i2i = extra_image_feat @ image_feat_all / self.extra_temp 
                extra_sim_t2t = extra_text_feat @ text_feat_all / self.extra_temp

                loss_extra_i2t = -torch.sum(F.log_softmax(extra_sim_i2t, dim=1)*extra_i2t_targets,dim=1).mean()
                loss_extra_t2i = -torch.sum(F.log_softmax(extra_sim_t2i, dim=1)*extra_t2i_targets,dim=1).mean()
                loss_extra_i2i = -torch.sum(F.log_softmax(extra_sim_i2i, dim=1)*extra_i2i_targets,dim=1).mean()
                loss_extra_t2t = -torch.sum(F.log_softmax(extra_sim_t2t, dim=1)*extra_t2t_targets,dim=1).mean()

                loss_cncl = (loss_extra_i2t+loss_extra_t2i+loss_extra_i2i+loss_extra_t2t)/4
                

                del extra_t2t_targets, extra_i2i_targets, extra_t2i_targets, extra_i2t_targets, extra_text_targets, extra_img_targets
                del extra_sim_t2t_m, extra_sim_i2i_m, extra_sim_t2i_m, extra_sim_i2t_m, extra_text_feat_m, extra_text_output_m
                del text_feat_all, text_output_m, extra_image_feat_m, image_feat_all, image_embeds_m, extra_img, extra_text
            else:
                with torch.no_grad():
                    self._momentum_update(self.extra_model_pairs)
                    batch_size = image.shape[0]
                    
                    image_feat_m_extra = self.extra_image_queue.clone().detach().permute(1, 0)[:batch_size, :].contiguous().to(image.device)
                    text_feat_m_extra = self.extra_text_queue.clone().detach().permute(1, 0)[:batch_size, :].contiguous().to(image.device)

            self.extra_dequeue_and_enqueue(image_feat_m_extra, text_feat_m_extra)

            ##############====== CAP-img-fusion ======################
            image_embeds_origpair = self.visual_encoder(image)
            text_output_origpair = self.text_encoder.bert(text_orig.input_ids, attention_mask = text_orig.attention_mask, return_dict = True, mode = 'text')
            text_embeds_origpair = text_output_origpair.last_hidden_state

            ##================= MAC ========================##
            image_feat = F.normalize(self.vision_proj(image_embeds_origpair[:,0,:]),dim=-1)
            text_feat = F.normalize(self.text_proj(text_embeds_origpair[:,0,:]),dim=-1)                 
                
            # get momentum features
            with torch.no_grad():
                self._momentum_update(self.model_pairs)
                image_embeds_m = self.visual_encoder_m(image) 
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)           

                text_output_m = self.text_encoder_m.bert(text_orig.input_ids, attention_mask = text_orig.attention_mask, return_dict = True, mode = 'text')    
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                # only orig should be aligned, 1 here means img-text aligned 
                sim_targets[real_label_pos, real_label_pos] = 1 

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets_g2g.fill_diagonal_(1)       
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            sim_i2t = image_feat @ text_feat_all / self.temp 
            sim_t2i = text_feat @ image_feat_all / self.temp 
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
            
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_g2g,dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_g2g,dim=1).mean()

            loss_MAC = (loss_i2t+loss_t2i)/4

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            del sim_t2i_targets, sim_i2t_targets, sim_targets_g2g, sim_targets, sim_t2i_m, sim_i2t_m, text_feat_all, text_output_m, image_feat_all
            del text_feat, image_feat, sim_i2i, sim_t2t, sim_i2t, sim_t2i
            ##================= Loss_2cls ========================## 
            # forward the positve image-text pair
            if len(image_indices) > 0 :
                image_embeds = self.extra_imgs_fusion(image_embeds_origpair, extra_image_embeds, image_indices)
                del extra_image_embeds
            else:
                image_embeds = image_embeds_origpair
            image_atts = torch.ones(image_embeds_origpair.size()[:-1],dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            del text_output
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )            
            with torch.no_grad():
                bs = image.size(0)          

            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
            itm_labels[real_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
            vl_output = self.itm_head(output_pos.last_hidden_state[:,0,:])   
            loss_2cls = F.cross_entropy(vl_output, itm_labels) 
            del vl_output, itm_labels, real_label_pos
            ##================= Loss_mcls ========================## 
            output_cls = self.cls_head(output_pos.last_hidden_state[:,0,:])
            loss_mcls = F.binary_cross_entropy_with_logits(output_cls, multicls_label.type(torch.float))
            del output_cls, multicls_label
            ##================= Loss_pat ========================## 
            image_embeds_patchcls = self.patch_2dec(image_embeds_origpair[:,1:,:])
            patch_logits = self.patch_2cls_head(image_embeds_patchcls).view(-1,2)
            patch_label = patch_label.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss_pat = loss_fct(patch_logits, patch_label).mean()
            image_embeds_patchcls = self.patch_proj(image_embeds_patchcls)
            del patch_logits, patch_label
            ##================= Loss_bbox and Loss_giou ========================## 
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
            binary_image = output_pos.last_hidden_state[:,0,:].unsqueeze(1).clone()
            binary_image = self.image_binary_proj(binary_image)
            binary_text = output_pos.last_hidden_state[:,0,:].unsqueeze(1).clone()
            binary_text = self.text_binary_proj(binary_text)
            del output_pos

            image_fusion_embedding = self.binary_dec(query=self.norm_layer_2_dec(binary_image),
                                               key=self.norm_layer_2_dec(image_embeds_patchcls),
                                               value=self.norm_layer_2_dec(image_embeds_patchcls))[0]
            
            cls_tokens_local = self.patch_dec(query = self.norm_layer_patch_dec(cls_tokens_local),
                                              key = self.norm_layer_patch_dec(image_fusion_embedding),
                                              value = self.norm_layer_patch_dec(image_fusion_embedding))[0]
            del image_embeds_patchcls
            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = image_embeds_origpair + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            del local_feat_it_cross_attn, local_feat_aggr, output_coord, image_embeds_origpair
            ##================= Loss_tok ========================##
            token_label = text_orig.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            token_label[token_label==0] = -100 # -100 index = padding token
            token_label[token_label==1] = 0
            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1

            input_ids = text_orig.input_ids.clone()

            if len(image_indices) > 0 :
                with torch.no_grad():
                    image_embeds_m = self.extra_imgs_fusion(image_embeds_m, extra_image_embeds_m, image_indices)

            with torch.no_grad():
                image_embeds_m = self.text_decoder_fusion(query = self.norm_layer_decoder_fusion(image_embeds_m),
                                                        key = self.norm_layer_decoder_fusion(binary_text),
                                                        value = self.norm_layer_decoder_fusion(binary_text))[0]
            
            image_embeds_fusion = self.text_decoder_fusion(query = self.norm_layer_decoder_fusion(image_embeds),
                                                        key = self.norm_layer_decoder_fusion(binary_text),
                                                        value = self.norm_layer_decoder_fusion(binary_text))[0]

            if self.args.token_momentum:
                with torch.no_grad():
                    logits_m = self.text_encoder_m(input_ids, 
                                                attention_mask = text_orig.attention_mask,
                                                encoder_hidden_states = image_embeds_m,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                return_logits = True,   
                                                )    
                token_cls_output = self.text_encoder(input_ids, 
                                            attention_mask = text_orig.attention_mask,
                                            encoder_hidden_states = image_embeds_fusion,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            soft_labels = F.softmax(logits_m.view(-1, 2),dim=-1),
                                            alpha = alpha
                                            )    
            else:
                token_cls_output  = self.text_encoder(input_ids, 
                                            attention_mask = text_orig.attention_mask,
                                            encoder_hidden_states = image_embeds_fusion,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            )  

            loss_tok = token_cls_output.loss

            return loss_MAC, loss_2cls, loss_bbox, loss_giou, loss_tok, loss_mcls, loss_cncl, loss_pat

        else:
            image_embeds_origpair = self.visual_encoder(image)
            if len(image_indices) > 0 :
                extra_image_embeds = self.visual_encoder(extra_img)
                image_embeds = self.extra_imgs_fusion(image_embeds_origpair, extra_image_embeds, image_indices)
            else:
                image_embeds = image_embeds_origpair
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            text_output_origpair = self.text_encoder.bert(text_orig.input_ids, attention_mask = text_orig.attention_mask, return_dict = True, mode = 'text')            
            text_embeds_origpair = text_output_origpair.last_hidden_state

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state

            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )               
            ##================= bbox ========================## 
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
            image_embeds_patchcls = self.patch_2dec(image_embeds_origpair[:,1:,:])
            image_embeds_patchcls = self.patch_proj(image_embeds_patchcls)
            binary_image = output_pos.last_hidden_state[:,0,:].unsqueeze(1).clone()
            binary_image = self.image_binary_proj(binary_image)
            binary_text = output_pos.last_hidden_state[:,0,:].unsqueeze(1).clone()
            binary_text = self.text_binary_proj(binary_text)
            image_fusion_embedding = self.binary_dec(query=self.norm_layer_2_dec(binary_image),
                                               key=self.norm_layer_2_dec(image_embeds_patchcls),
                                               value=self.norm_layer_2_dec(image_embeds_patchcls))[0]
            
            cls_tokens_local = self.patch_dec(query = self.norm_layer_patch_dec(cls_tokens_local),
                                              key = self.norm_layer_patch_dec(image_fusion_embedding),
                                              value = self.norm_layer_patch_dec(image_fusion_embedding))[0]

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = image_embeds_origpair + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            ##================= 2cls ========================## 
            logits_real_fake = self.itm_head(output_pos.last_hidden_state[:,0,:])
            ##================= mcls ========================## 
            logits_multicls = self.cls_head(output_pos.last_hidden_state[:,0,:])
            ##================= tok_cls ========================##   
            input_ids = text_orig.input_ids.clone()
            image_embeds_fusion = self.text_decoder_fusion(query = self.norm_layer_decoder_fusion(image_embeds),
                                                        key = self.norm_layer_decoder_fusion(binary_text),
                                                        value = self.norm_layer_decoder_fusion(binary_text))[0]
            logits_tok = self.text_encoder(input_ids, 
                                        attention_mask = text_orig.attention_mask,
                                        encoder_hidden_states = image_embeds_fusion,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        return_logits = True,   
                                        )     
            return logits_real_fake, logits_multicls, output_coord, logits_tok

    def extra_imgs_fusion(self,image_embeds, extra_image_embeds, image_indices):

        batch_size, seq_length, embed_dim = image_embeds.shape
        device = image_embeds.device

        result_embeds = torch.zeros_like(image_embeds).to(device)

        for i in range(batch_size):
            if i not in image_indices:
                result_embeds[i] = image_embeds[i]
            else:
                extra_indices = [idx for idx, img_idx in enumerate(image_indices) if img_idx == i]
                extra_selected = extra_image_embeds[extra_indices]  

                extra_selected = extra_selected.view(1, -1, embed_dim)

                q = image_embeds[i].unsqueeze(0)
                k = v = extra_selected  

                attn_output, _ =  self.extraimgs_fusion(query=q, key=k, value=v)
                result_embeds[i] = attn_output.squeeze(0)

        return result_embeds   


    @torch.no_grad()    
    def copy_params(self, model_pairs):
        for model_pair in model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False


    @torch.no_grad()    
    def init_extra_proj(self, model_pairs):
        for model_pair in model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)    

            
    @torch.no_grad()        
    def _momentum_update(self, model_pairs):
        for model_pair in model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def extra_dequeue_and_enqueue(self, image_feat, text_feat):

        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.extra_queue_ptr)
        assert self.queue_size % batch_size == 0 

        self.extra_image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.extra_text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size ) % self.queue_size

        self.extra_queue_ptr[0] = ptr  
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def get_label(A, image_indices, if_source_name_img):

    image_indices = torch.tensor(image_indices)
    if_source_name_img = torch.tensor(if_source_name_img)

    rows_to_modify = torch.where(if_source_name_img == 0)[0]

    cols_to_modify = image_indices[rows_to_modify]

    A[rows_to_modify, cols_to_modify] = 1

    return A


