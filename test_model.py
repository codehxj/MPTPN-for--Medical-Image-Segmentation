import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT import LViT
from nets.LViT_decoder import LViT_decoder
from nets.LViT_encoder import LViT_encoder
from utils import *
import cv2
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def show_image_with_dice(predict_save, labs):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    acc_pred = accuracy_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    precision_pred, recall_pred, f_score_pred, _ = precision_recall_fscore_support(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))

    # if config.task_name == "MoNuSeg":
    #     predict_save = cv2.pyrUp(predict_save, (448, 448))
    #     predict_save = cv2.resize(predict_save, (2000, 2000))
    #     # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    #     # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
    #     cv2.imwrite(save_path, predict_save * 255)
    # else:
    #     cv2.imwrite(save_path, predict_save * 255)

    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred, acc_pred, precision_pred, recall_pred, f_score_pred

def vis_and_save_heatmap1(preds, labs):
    output = preds
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    labs = labs.cpu().data.numpy()
    dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp = show_image_with_dice(predict_save, labs)
    return dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp


def vis_and_save_heatmap(model1, model2, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model2.eval()

    x1, x2, x3 = model1(input_img.cuda(), text.cuda())  # [2,1,256,256]  ！！！！！！[2,1,224,224]
    output = model2(x1, x2, x3)
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp = show_image_with_dice(predict_save, labs)
    # output = output.flatten()
    # labs = labs.flatten()
    # tn, fp, fn, tp = confusion_matrix(labs, output, labels=[0, 1]).ravel()
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    return dice_pred_tmp, iou_tmp, acc_tmp, precision_tmp, recall_tmp, f_score_tmp

def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    #intersection = np.sum(intersection)
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    iou = (intersection + 1e-15) / (union + 1e-15)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return iou, precision, recall

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    if config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Covid19":
        test_num = 8
        model_type = config.model_name
        model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Glas":
        test_num = 11
        model_type = config.model_name
        model_path = "./Glas/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        # model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        model1 = LViT_encoder(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        model1.load_state_dict(torch.load("D:/deep_learning3/simsiam-main/checkpoint_natural/checkpoint_0004.pt"), False)
        print('Load successful!')
        model2 = LViT_decoder(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    else:
        raise TypeError('Please enter a valid name for the model type')

    model1 = model1.cuda()
    model2 = model2.cuda()
    #model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.load_state_dict(checkpoint.state_dict(), strict=False)
    model2.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    acc_pred = 0.0
    dice_ens = 0.0
    precision_pred = 0.0
    recall_pred = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            test_data, test_label, test_text = test_data.cuda(), test_label.cuda(), test_text.cuda()
            x1, x2, x3 = model1(test_data, test_text)  # [2,1,256,256]  ！！！！！！[2,1,224,224]
            preds = model2(x1, x2, x3)  # [2,1,224,224]
            dice_pred_t, iou_pred_t, acc_pred_t, precision_t, recall_t, f_score_t = vis_and_save_heatmap1(preds, test_label)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            acc_pred += acc_pred_t
            precision_pred += precision_t
            recall_pred += recall_t
            torch.cuda.empty_cache()
            pbar.update()
            print("name:{}   dice:{}   iou:{}".format(names, dice_pred_t, iou_pred_t))
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
    print("acc_pred", acc_pred / test_num)
    print("precision", precision_pred / test_num)
    print("recall", recall_pred / test_num)
