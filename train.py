# -*- coding: utf-8 -*-
# @Time    : 2/9/23 8:22 PM
# @Author  : LIANYONGXING
# @FileName: train.py
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from wrapper import CLIPWrapper
from text_image_dataset import image_title_dataset
from torch.utils.data import DataLoader

def main(hparams):

    model = CLIPWrapper()
    list_image_path = ['images/app.jpg', 'images/paper.jpg']
    list_txt = ['a picture of an app', 'description for paper']
    dataset = image_title_dataset(list_image_path, list_txt)

    train_dataloader = DataLoader(dataset, batch_size=1)  # Define your own dataloader
    trainer = Trainer.from_argparse_args(hparams, precision='bf16', max_epochs=2)

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
    print("finish!")