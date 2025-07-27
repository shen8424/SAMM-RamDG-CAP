import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.dataset import SAMM
from dataset.randaugment import RandomAugment

def create_dataset(config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])    
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    
    train_dataset = SAMM(config=config, ann_file=config['train_file'], transform=train_transform, max_words=config['max_words'], is_train=True)              
    val_dataset = SAMM(config=config, ann_file=config['val_file'], transform=test_transform, max_words=config['max_words'], is_train=False)              
    return train_dataset, val_dataset    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders

def collate_fn(batch):

    image = [item[0] for item in batch]
    label = [item[1] for item in batch]
    caption = [item[2] for item in batch]
    fake_image_box = [item[3] for item in batch]
    fake_text_pos_list = [item[4] for item in batch]
    W = [item[5] for item in batch]
    H = [item[6] for item in batch]
    cap_images = [item[7] for item in batch]
    if_source_name_img = [item[8] for item in batch]
    extra_text = [item[9] for item in batch]
    if_source_name_text = [item[10] for item in batch]
    patch_label = [item[11] for item in batch]


    stacked_imgs_list = []  
    image_indices = []
    text_indices = [] 

    for idx, images in enumerate(cap_images):
        if len(images) > 0:
            stacked_imgs_list.extend(images)
            image_indices.extend([idx] * len(images))


    if len(stacked_imgs_list) > 0:
        cap_images = torch.stack(stacked_imgs_list, dim=0)
    else:
        cap_images = torch.tensor([])

    flattened_list_img = []
    flattened_list_text = []

    for sublist in if_source_name_img:
        if sublist:  
            flattened_list_img.extend(sublist)

    for sublist in if_source_name_text:
        if sublist:  
            flattened_list_text.extend(sublist)
    
    cap_texts = []

    for idx, texts in enumerate(extra_text):
        if len(texts) > 0:
            cap_texts.extend(texts)
            text_indices.extend([idx] * len(texts))

    image = torch.stack(image, dim=0)
    fake_image_box = torch.stack(fake_image_box, dim=0)
    fake_text_pos = torch.stack(fake_text_pos_list, dim=0)
    patch_label = torch.stack(patch_label, dim=0)

    return image, label, caption, fake_image_box, fake_text_pos, W, H, cap_images, image_indices, flattened_list_img, cap_texts, text_indices, flattened_list_text, patch_label
