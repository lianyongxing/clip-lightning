# -*- coding: utf-8 -*-
# @Time    : 2/9/23 7:47 PM
# @Author  : LIANYONGXING
# @FileName: wrapper.py
import pytorch_lightning as pl
import clip
import torch.optim as optim
import torch
import torch.nn as nn


class CLIPWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32",jit=False)
        self.batch_size = 1
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def training_step(self, batch, idx):
        optimizer = self.optimizers().optimizer

        optimizer.zero_grad()

        images, texts = batch
        logits_per_image, logits_per_text = self.model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long)
        total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        optimizer.step()

    def validation_step(self, batch, idx):
        images, texts = batch
        image_logits, text_logits = self.forward(images, texts)
        ground_truth = torch.arange(len(image_logits))
        loss = (self.loss_img(image_logits, ground_truth) + self.loss_txt(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        return {'optimizer': optimizer}

    def forward(self, text=None, image=None):
        if (text is None) and (image is None):
            raise ValueError("Provide either text or image")

        elif image is not None:
            x = self.model.encode_image(image)
            x /= x.norm(dim=-1, keepdim=True)
            return x

        elif text is not None:
            x = self.model.encode_text(text)
            x /= x.norm(dim=-1, keepdim=True)
            return x