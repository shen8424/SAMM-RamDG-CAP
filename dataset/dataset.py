from distutils.command.config import config
import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

class SAMM(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = config["images_file"]       
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']
        image_dir_all = f"{self.root_dir}/{img_dir}"
        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
        
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        
        patch_label = [0] * 256
        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            # calculate patch_label (FVRM)
            result_x = []
            result_y = []
            for k in range(1, 17):
                value = 16 * k / 256
                if x <= value <= x + w: 
                    result_x.append(k)
            for k in range(1, 17):
                value = 16 * k / 256 
                if y <= value <= y + h: 
                    result_y.append(k)

            indices = [(y - 1) * 16 + (x - 1) for y in result_y for x in result_x]

            for index in indices:
                patch_label[index] = 1
            
            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        
        cap_images = []
        idx_cap_images = []
        idx_cap_texts = []
        cap_texts = []

        if self.is_train:
            if ann.get("cap_images"):
                for key, folder_path in ann["cap_images"].items():
                    try:
                        selected_image_name = f"{random.randint(0, 2)}.jpg"
                        selected_image_path = f"{self.root_dir}/people_imgs/{folder_path}/{selected_image_name}"
                        
                        extra_image = Image.open(selected_image_path).convert('RGB')
                        
                        if self.is_train:
                            if rand() < 0.5:
                                extra_image = hflip(extra_image)
                            
                            extra_image = resize(extra_image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
                        
                        extra_image = self.transform(extra_image)
                        cap_images.append(extra_image)
                
                    except Exception as e:
                        raise ValueError(f"### Warning: Failed to process image {selected_image_path}. Error: {e}")
        else:
            if ann.get("cap_images"):
                for key, folder_path in ann["cap_images"].items():
                    try:
                        selected_image_name = f"{0}.jpg"
                        selected_image_path = f"{self.root_dir}/people_imgs/{folder_path}/{selected_image_name}"
                        
                        extra_image = Image.open(selected_image_path).convert('RGB')
                        
                        if self.is_train:
                            if rand() < 0.5:
                                extra_image = hflip(extra_image)
                            
                            extra_image = resize(extra_image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
                        
                        extra_image = self.transform(extra_image)
                        cap_images.append(extra_image)
                
                    except Exception as e:
                        raise ValueError(f"### Warning: Failed to process image {selected_image_path}. Error: {e}")
        
        if ann.get("idx_cap_images"):
            idx_cap_images = ann["idx_cap_images"]
        if ann.get("idx_cap_texts"):
            idx_cap_texts = ann["idx_cap_texts"]
        
        if ann.get("cap_texts"):
            for key, value in ann["cap_texts"].items():
                value = pre_caption(value, self.max_words)
                cap_texts.append(value)
        
        patch_label = torch.tensor(patch_label)

        return image, label, caption, fake_image_box, fake_text_pos_list, W, H, cap_images, idx_cap_images, cap_texts, idx_cap_texts, patch_label
