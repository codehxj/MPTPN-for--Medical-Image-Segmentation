# -*- coding: utf-8 -*-
import PIL.Image
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from bert_embedding import BertEmbedding
import clip


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, text = sample['image'], sample['text']
        image[0], image[1] = image[0].numpy(), image[1].numpy()
        image[0], image[1] = image[0].astype(np.uint8), image[1].astype(np.uint8)  # OSIC
        image[0], image[1] = F.to_pil_image(image[0]), F.to_pil_image(image[1])
        x, y = image[0].size

        if x != self.output_size[0] or y != self.output_size[1]:
            image[0] = zoom(image[0], (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            image[1] = zoom(image[1], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image[0] = F.to_tensor(image[0])
        image[1] = F.to_tensor(image[1])
        #label = to_long_tensor(label)
        text = torch.Tensor(text)
        imagelist = []
        imagelist.append(image[0])
        imagelist.append((image[1]))
        #sample = {'image': imagelist, 'label': label, 'text': text}
        sample = {'image': imagelist, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, text = sample['image'], sample['text']
        image[0], image[1] = image[0].astype(np.uint8), image[1].astype(np.uint8)  # OSIC
        image[0], image[1] = F.to_pil_image(image[0]), F.to_pil_image(image[1])
        x, y = image[0].size
        if x != self.output_size[0] or y != self.output_size[1]:
            image[0] = zoom(image[0], (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            image[1] = zoom(image[1], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image[0] = F.to_tensor(image[0])
        image[1] = F.to_tensor(image[1])
        #label = to_long_tensor(label)
        text = torch.Tensor(text)
        imagelist = []
        imagelist.append(image[0])
        imagelist.append((image[1]))
        #sample = {'image': imagelist, 'label': label, 'text': text}
        sample = {'image': imagelist, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            img = img.permute(1,2,0)
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        # mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        # mask = cv2.resize(mask, (self.image_size, self.image_size))
        # mask[mask <= 0] = 0
        # mask[mask > 0] = 1
        # mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 14:
            text = text[:14, :]
        sample = {'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None, train_imgtf: Callable = None, one_hot_mask: int = False, image_size: int = 224) -> None:

        self.dataset_path = dataset_path
        self.image_size = image_size
        #self.input_path = os.path.join(dataset_path, 'img')
        #self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.dataset_path)
        #self.mask_list = os.listdir(self.output_path)

        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.train_imgtf = train_imgtf

    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]  # MoNuSeg
        mask_filename = image_filename[: -3] + "jpg"  # MoNuSeg
        # mask_filename = self.mask_list[idx]  # Covid19
        # image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = PIL.Image.open(os.path.join(self.dataset_path, image_filename))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size))

        #=======================TwoTransform=======================
        imageAuglist = []
        for i in range(16):
            img = self.train_imgtf(image)
            imageAuglist.append(img)
        image1 = random.choice(imageAuglist)
        image2 = random.choice(imageAuglist)
        #===========================================================
        # read mask image
        # mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        # mask = cv2.resize(mask, (self.image_size, self.image_size)) #(224,224)
        # mask[mask <= 0] = 0
        # mask[mask > 0] = 1

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

        # correct dimensions if needed
        #==============源代码=========================================================
        #image, mask = correct_dims(image, mask) #image(224,224,3)  mask(224,224,1)
        #============================================================================

        # ==============图像增强为两张图片之后的修正维度====================================
        imagelist2 = []
        # imagelist[0], mask = correct_dims(imagelist[0], mask)  # image(224,224,3)  mask(224,224,1)
        # imagelist[1]  = correct_dims(imagelist[1])
        #image1, mask = correct_dims(image1, mask)  # image(224,224,3)  mask(224,224,1)
        image1 = correct_dims(image1)  # image(224,224,3)  mask(224,224,1)
        image2 = correct_dims(image2)
        imagelist2.append(image1)
        imagelist2.append(image2)
        #==============================================================================

        text = self.rowtext[mask_filename] #是一个字符串
        #text_clip = clip.tokenize(text) #1,77
        text = text.split('\n') #把字符串放进列表
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 10:
           text = text[:10, :]#[10,768]
        else:
            print(mask_filename)

        # if self.one_hot_mask:
        #     assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
        #     mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # ==============源代码============================================================
        #sample = {'image': image, 'label': mask, 'text': text, 'text_clip': text_clip}
        #================================================================================

        # ==============两张图片的返回值======================================================
        #sample = {'image': imagelist2, 'label': mask, 'text': text}
        sample = {'image': imagelist2, 'text': text}
        # =================================================================================

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
