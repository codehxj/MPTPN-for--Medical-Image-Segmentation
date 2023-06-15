# -*- coding: utf-8 -*-
import argparse

import torch.optim
import torch.nn as nn
import time

from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
import joblib
from torch.backends import cudnn

from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from nets.LViT import LViT
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary, train_one_epoch2
import Config as config
from torchvision import transforms

from nets.LViT_decoder import LViT_decoder
from nets.LViT_encoder import LViT_encoder
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, CosineSimilarityLoss,WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    test_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf, image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf,
                                      image_size=config.img_size)

    elif config.task_name == 'Glas':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf, image_size=config.img_size)

    elif config.task_name == 'CoSKEL':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf, image_size=config.img_size)

    elif config.task_name == 'FruitFlower':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=worker_init_fn, num_workers=0, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=worker_init_fn, num_workers=0, pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)

    checkpoint = torch.load("D:/deep_learning3/simsiam-main/checkpoint_natural3/checkpoint_0004.pth")
    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model1 = LViT_encoder(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        #model1.load_state_dict(torch.load("D:/deep_learning3/simsiam-main/checkpoint_natural/checkpoint_0004.pt"), False)
        logger.info('Load successful!')
        model2 = LViT_decoder(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        #model2.load_state_dict(torch.load("D:/deep_learning3/LViT2-main/MoNuSeg/LViT/Test_session_10.21_16h05/models/best_model-LViT.pt"), False)


    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        pretrained_UNet_model_path = "MoNuSeg/LViT_pretrain/Test_session_10.07_16h01/models/best_model-LViT_pretrain.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')

    model1 = model1.cuda()
    model2 = model2.cuda()

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr)  # Choose optimize

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None


    max_dice = 0.0
    max_precision = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model2.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model1, model2, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup

        # ==================evaluate on test set==========================================
        logger.info('Test')
        with torch.no_grad():
            model2.eval()
            test_loss, test_iou, test_acc, test_dice, test_precision = train_one_epoch2(test_loader, model1, model2, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =================================================================================
        # =============================================================
        #       Save best model
        # =============================================================
        # ================ON test dataset dice====================================
        if test_dice > max_dice:
            if epoch + 1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, test_dice))
                max_dice = test_dice
                best_epoch = epoch + 1
                if not os.path.isdir(config.model_path):
                    os.makedirs(config.model_path)
                #torch.save(model2.state_dict(), config.model_path+ '/'+'best_model-LViT.pt')
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model2.state_dict(),
                                 'optimizer': optimizer.state_dict()}, config.model_path)
                #torch.save(model.state_dict(), './checkpoint_natural/checkpoint_{:04d}.pt'.format(epoch))
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, the best is still: {:.4f} in epoch {}'.format(test_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))
        # =============================================================================================================

        # ================ON test dataset precision====================================
        # if test_precision > max_precision:
        #     if epoch + 1 > 5:
        #         logger.info(
        #             '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_precision, test_precision))
        #         max_precision = test_precision
        #         best_epoch = epoch + 1
        #         # save_checkpoint({'epoch': epoch, 'best_model': True, 'model': model_type, 'state_dict': model2.state_dict(),
        #         #                  'test_loss': test_loss, 'optimizer': optimizer.state_dict()}, config.model_path)
        #         if not os.path.isdir(config.model_path):
        #             os.makedirs(config.model_path)
        #         torch.save(model2.state_dict(), config.model_path + '/' + 'best_model-LViT.pt')
        #         # torch.save(model.state_dict(), './checkpoint_natural/checkpoint_{:04d}.pt'.format(epoch))
        # else:
        #     logger.info('\t Mean precision:{:.4f} does not increase, the best is still: {:.4f} in epoch {}'.format(test_precision,
        #                                                                                                       max_precision,
        #                                                                                                       best_epoch))
        # early_stopping_count = epoch - best_epoch + 1
        # logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))
        # =============================================================================================================

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break
        # =======================================================================
    return model2


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
