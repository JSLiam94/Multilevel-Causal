import os
import sys
import argparse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from datagen_cls import ListDataset
import random
import numpy as np
from tqdm import tqdm
from src_u.loss_functions.losses import create_loss
from src_u.utils import mAP, ModelEma, add_weight_decay
from src_u.models import IDA33u



locaa=time.strftime('%Y-%m-%d-%H-%M-%S')


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', help='(coco,voc,nuswide)', default='coco')
parser.add_argument('--data_path', default='',help='path to dataset')
parser.add_argument('--transforms', help='data transform style (asl or mlgacn)', default='asl')
#parser.add_argument('--pretrain_path', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth')
parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--num-classes', default=8)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=512, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--save_path',default='./save')
parser.add_argument('--loss', default='mlsm', type=str, help='(mlsm,bce,focal,asl,halfasl)')
parser.add_argument('--use_intervention', default=True, type=bool)
parser.add_argument('--heavy', default=False, type=bool)
parser.add_argument('--backbone', default='resvit32fu', type=str, help='(resnet101,swim_transformer,swim_transformer_large,transformer)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--seed', type=int, default=2025)

def main():
    
    args = parser.parse_args(sys.argv[1:])

    if args.seed is not None: 
        seed = args.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    print('creating model...')
    model = IDA33u(backbone=args.backbone, num_classes=args.num_classes, pretrain=args.pretrain_path, use_intervention=args.use_intervention, heavy=args.heavy)
    args.feat_dim = model.feat_dim
    model = model.cuda()
    print('done\n')
    
    train_dataset = ListDataset(root='D:/SUCAI/OIA-ODIR/Training-Set/Images',
                           list_file='D:\SUCAI\OIA-ODIR\Training-Set\Annotation/train.csv',input_size=args.input_size, state='Train')
    val_dataset = ListDataset(root='D:/SUCAI/OIA-ODIR/On-site/Images',
                           list_file='D:\SUCAI\OIA-ODIR\On-site\Annotation/anno.csv',input_size=args.input_size, state='Valid')

    #data_loader_train = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=False, num_workers=4,drop_last=True)

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # Actuall Training
    model = train_multi_label_coco(model, train_loader, val_loader, args)
    

def train_multi_label_coco(model, train_loader, val_loader, args):
    #base_lr = 4e-4 * args.batch_size /128

    base_lr = args.lr
    save_name = args.save_path
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 100
    Stop_epoch = 100
    weight_decay = 1e-4

    criterion = create_loss(args.loss)    
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=base_lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    #!
    if not os.path.exists(save_name):
        os.mkdir(save_name)

    highest_mAP = 0

    scaler = GradScaler()

    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Epochs}")
        for i, (inputData, target) in enumerate(train_bar):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            # targets_one_hot = torch.nn.functional.one_hot(target[:,1].to(torch.int64), num_classes=6)
            # target = torch.cat([target[:,0][:, None],targets_one_hot],dim=1)
            with autocast():  # mixed precision
                output = model(inputData)  # sigmoid will be done in loss 
                output = output.float()
                loss = criterion(output, target)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            # store information
            # if i % 100 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
            #           .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
            #                   scheduler.get_last_lr()[0], \
            #                   loss.item()))
            train_bar.desc = f'Epoch [{epoch + 1}/{Epochs}], Step [{i + 1}/{steps_per_epoch}], LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.1f}'
        #_, mask =  feat
        #att_mask, dropout_mask = mask
        #print (att_mask)
        #print (dropout_mask)
        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(save_name, 'model-highest_model.ckpt'))
            try:
                torch.save({'model':ema.module.state_dict(), 'epoch': epoch}, os.path.join(save_name, 'model-highest.ckpt'))
            except:
                pass
        
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
    torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(save_name, 'model-last_model.ckpt'))
        

import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda import amp

def validate_multi(val_loader, model, ema_model, class_names, save_dir):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    val_bar = tqdm(val_loader, desc='validating')
    
    # 用于保存每个类别的指标
    class_acc = {name: [] for name in class_names}
    class_pre = {name: [] for name in class_names}
    class_recall = {name: [] for name in class_names}
    
    for i, (input, target) in enumerate(val_bar):
        target = target.cuda()  # (batch,3,num_classes)
        
        with torch.no_grad():
            with amp.autocast():
                out_m = model(input.cuda())
                ema_out = ema_model.module(input.cuda())
                output_regular = Sig(out_m).cpu()
                output_ema = Sig(ema_out).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    targets = torch.cat(targets).numpy()
    preds_regular = torch.cat(preds_regular).numpy()
    preds_ema = torch.cat(preds_ema).numpy()
    
    # 计算指标
    # mAP
    mAP_score_regular = mAP(targets, preds_regular)
    mAP_score_ema = mAP(targets, preds_ema)
    
    # 准确率
    acc_regular = accuracy_score(targets.argmax(axis=1), preds_regular.argmax(axis=1))
    acc_ema = accuracy_score(targets.argmax(axis=1), preds_ema.argmax(axis=1))
    
    # 平衡准确率
    bacc_regular = balanced_accuracy_score(targets.argmax(axis=1), preds_regular.argmax(axis=1))
    bacc_ema = balanced_accuracy_score(targets.argmax(axis=1), preds_ema.argmax(axis=1))
    
    # AUC-ROC
    auroc_regular = roc_auc_score(targets, preds_regular)
    auroc_ema = roc_auc_score(targets, preds_ema)
    
    # Cohen's Kappa
    kappa_regular = cohen_kappa_score(targets.argmax(axis=1), preds_regular.argmax(axis=1))
    kappa_ema = cohen_kappa_score(targets.argmax(axis=1), preds_ema.argmax(axis=1))
    
    # F1分数
    f1_regular = f1_score(targets.argmax(axis=1), preds_regular.argmax(axis=1), average='macro')
    f1_ema = f1_score(targets.argmax(axis=1), preds_ema.argmax(axis=1), average='macro')
    
    # 每个类别的指标
    for i, name in enumerate(class_names):
        # 准确率
        class_acc[name].append(accuracy_score(targets[:,i], preds_regular[:,i] > 0.5))
        # 精确率
        class_pre[name].append(precision_score(targets[:,i], preds_regular[:,i] > 0.5))
        # 召回率
        class_recall[name].append(recall_score(targets[:,i], preds_regular[:,i] > 0.5))
    
    # 绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], preds_regular[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制ROC曲线
    plt.figure()
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/roc_curve.png')
    plt.close()
    
    # 保存结果到TXT文件
    with open(save_dir + '/results.txt', 'w') as f:
        f.write("Validation Results:\n")
        f.write("mAP: {:.2f}, mAP score EMA: {:.2f}\n".format(mAP_score_regular, mAP_score_ema))
        f.write("Acc: {:.2f}, Accuracy EMA: {:.2f}\n".format(acc_regular, acc_ema))
        f.write("Bacc: {:.2f}, Balanced Accuracy EMA: {:.2f}\n".format(bacc_regular, bacc_ema))
        f.write("AUROC: {:.2f}, AUROC EMA: {:.2f}\n".format(auroc_regular, auroc_ema))
        f.write("Kappa: {:.2f}, Cohen's Kappa EMA: {:.2f}\n".format(kappa_regular, kappa_ema))
        f.write("F1: {:.2f}, F1 Score EMA: {:.2f}\n".format(f1_regular, f1_ema))
        f.write("\nClass-wise Metrics:\n")
        for name in class_names:
            f.write("Class {}: Accuracy {:.2f}, Precision {:.2f}, Recall {:.2f}\n".format(
                name, np.mean(class_acc[name]), np.mean(class_pre[name]), np.mean(class_recall[name])))
    
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    print("Accuracy regular {:.2f}, Accuracy EMA {:.2f}".format(acc_regular, acc_ema))
    print("Balanced Accuracy regular {:.2f}, Balanced Accuracy EMA {:.2f}".format(bacc_regular, bacc_ema))
    print("AUROC regular {:.2f}, AUROC EMA {:.2f}".format(auroc_regular, auroc_ema))
    print("Cohen's Kappa regular {:.2f}, Cohen's Kappa EMA {:.2f}".format(kappa_regular, kappa_ema))
    print("F1 Score regular {:.2f}, F1 Score EMA {:.2f}".format(f1_regular, f1_ema))
    
    return max(mAP_score_regular, mAP_score_ema)



if __name__ == '__main__':
    main()