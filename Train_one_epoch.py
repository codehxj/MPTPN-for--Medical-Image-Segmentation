# -*- coding: utf-8 -*-

import torch.optim
import os
import time
from utils import *
import Config as config
import clip
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

device = torch.device('cuda')

def print_summary(name, epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, precision_t, train_precision_average,
                          recall_t, train_recall_average, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    string += 'Acc:{:.3f} '.format(acc)
    string += '(Avg {:.4f}) '.format(average_acc)
    string += 'precision:{:.3f}'.format(precision_t)
    string += '(Avg {:.4f})'.format(train_precision_average)
    string += 'recall:{:.3f}'.format(recall_t)
    string += '(Avg {:.4f})'.format(train_recall_average)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    string += 'Pic:{}'.format(name)
    summary += string
    logger.info(summary)
    # print summary

def print_summary2(name, iou, dice, logger):
    '''
        mode = Train or Test
    '''
    string = ''
    string += 'name:{} '.format(name)
    string += 'IoU:{:.3f} '.format(iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += 'Pic:{}'.format(name)
    summary += string
    logger.info(summary)

def vis_and_save_heatmap(preds, labs):
    output = preds
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    labs = labs.cpu().data.numpy()
    dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp = show_image_with_dice(predict_save, labs)
    return dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp

##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model1, model2, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model2.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    precision_pred = 0.0
    recall_pred = 0.0
    dices = []

    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        if text.shape[1] > 10:
            text = text[ :, :10, :]
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()

        # ====================================================
        #             Compute loss
        # ====================================================
        x1, x2, x3 = model1(images, text) #[2,1,256,256]  ！！！！！！[2,1,224,224]
        preds = model2(x1, x2, x3)#[2,1,224,224]
        out_loss = criterion(preds, masks.float())#masks[1,224,224]

        if model2.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)
        train_acc = acc_on_batch(masks, preds)
        dice_pred_t, iou_pred_t, acc_pred_t, precision_t, recall_t, f_score_t = vis_and_save_heatmap(preds, masks)

        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)


        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        precision_pred += precision_t.max()
        recall_pred += recall_t.max()

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
            train_precision_average = precision_pred / (config.batch_size*(i-1) + len(images))
            train_recall_average = recall_pred / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)
            train_precision_average = precision_pred / (config.batch_size * (i - 1) + len(images))
            train_recall_average = recall_pred / (config.batch_size * (i - 1) + len(images))

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(names, epoch + 1, i, len(loader), out_loss.item(), loss_name, batch_time,
                          average_loss.item(), average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, train_acc, train_acc_average, precision_t.max(), train_precision_average,
                          recall_t.max(), train_recall_average, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_iou_average, train_acc_average, train_dice_avg, train_precision_average


def train_one_epoch2(loader, model1, model2, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model2.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    precision_pred = 0.0
    recall_pred = 0.0
    dices = []

    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        if text.shape[1] > 10:
            text = text[ :, :10, :]

        images, masks, text = images.cuda(), masks.cuda(), text.cuda()

        x1, x2, x3 = model1(images, text) #[2,1,256,256]  ！！！！！！[2,1,224,224]
        preds = model2(x1, x2, x3)#[2,1,224,224]
        out_loss = criterion(preds, masks.float())#masks[1,224,224]

        if model2.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)
        train_acc = acc_on_batch(masks, preds)
        dice_pred_t, iou_pred_t, acc_pred_t, precision_t, recall_t, f_score_t = vis_and_save_heatmap(preds, masks)
        print("======================================")
        print("name:{}   dice:{}   iou:{}".format(names, train_dice, train_iou))


        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)


        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        precision_pred += precision_t.max()
        recall_pred += recall_t.max()

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
            train_precision_average = precision_pred / (config.batch_size*(i-1) + len(images))
            train_recall_average = recall_pred / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)
            train_precision_average = precision_pred / (config.batch_size * (i - 1) + len(images))
            train_recall_average = recall_pred / (config.batch_size * (i - 1) + len(images))

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(names, epoch + 1, i, len(loader), out_loss.item(), loss_name, batch_time,
                          average_loss.item(), average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, train_acc, train_acc_average, precision_t.max(), train_precision_average,
                          recall_t.max(), train_recall_average, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_iou_average, train_acc_average, train_dice_avg, train_precision_average