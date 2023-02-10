# -*- coding: utf-8 -*-
# @Time    : 2/9/23 7:35 PM
# @Author  : LIANYONGXING
# @FileName: text_image_dataset.py
from torch.utils.data import Dataset
import clip
import torch
from PIL import Image


device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training


class image_title_dataset(Dataset):
    def __init__(self, image_paths, texts):

        self.image_paths = image_paths
        self.texts = texts
        self.texts_token = clip.tokenize(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx])) # Image from PIL module
        text_token = self.texts_token[idx]
        return image, text_token



