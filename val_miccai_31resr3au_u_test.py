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

from src_u.loss_functions.losses import create_loss
from src_u.utils import mAP, ModelEma, add_weight_decay
from src_u.models import IDA33u
from sklearn.metrics import average_precision_score,precision_score,accuracy_score,recall_score,f1_score,hamming_loss,classification_report
from tqdm import tqdm


locaa=time.strftime('%Y-%m-%d-%H-%M-%S')


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', help='(coco,voc,nuswide)', default='coco')
parser.add_argument('--data_path', help='path to dataset')
parser.add_argument('--transforms', help='data transform style (asl or mlgacn)', default='asl')
#parser.add_argument('--pretrain_path', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth')
parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--num-classes', default=14)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--save_path',default='miccai_u/savetest')
parser.add_argument('--loss', default='mlsm', type=str, help='(mlsm,bce,focal,asl,halfasl)')
parser.add_argument('--use_intervention', default=True, type=bool)
parser.add_argument('--heavy', default=False, type=bool)
parser.add_argument('--backbone', default='resvit32fu', type=str, help='(resnet101,swim_transformer,swim_transformer_large,transformer)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--seed', type=int, default=1)

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
    
    state = torch.load('miccai_u/save/model-highest.ckpt')   
    print('creating model...')
    model = IDA33u(backbone=args.backbone, num_classes=args.num_classes, pretrain=None, use_intervention=args.use_intervention, heavy=args.heavy)
    args.feat_dim = model.feat_dim
    model.load_state_dict(state['model'])
    model = model.cuda()
    print('done\n')
    model = model.eval()

    test_dataset = ListDataset(root=['data_cxr_images'],
                           list_file='chestxray_spl/test_official.txt',input_size=224, state='Valid')

    #data_loader_train = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=False, num_workers=4,drop_last=True)

    # print("len(val_dataset)): ", len(val_dataset))
    # print("len(train_dataset)): ", len(train_dataset))
    print("len(test_dataset)): ", len(test_dataset))
    # Pytorch Data loader
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    # Actuall Training
    # model = train_multi_label_coco(model, train_loader, val_loader, args)
    validate_multi_change(test_loader, model)

def validate_multi_change(val_loader, model):
    Sig = torch.nn.Sigmoid()
    preds = []
    targets = []
    name_list = []
    predgt = []
    rename_pred = []
    rename_tar = []
    rename_tar2 = []
    eng_list = []


    for i, (input, target) in tqdm(enumerate(val_loader)):
        target = target
        # targets_one_hot = torch.nn.functional.one_hot(target[:,1].to(torch.int64), num_classes=6)
        # target = torch.cat([target[:,0][:, None],targets_one_hot],dim=1)
        # compute output
        with torch.no_grad():
            out_m = model(input.cuda())
            output = Sig(out_m).cpu()
            pred = output.gt(0.5).long()
    
        preds.append(output.detach())
        targets.append(target.cpu().detach())
        predgt.append(pred.detach())
        error_num = 0
        for ind in range(7):
            if int(pred[0][ind]) != int(target[0][ind]) :
                error_num+=1
        if error_num != 0:
            # name_list.append(fliename)
            str_pred = str(int(pred[0][0]))+'_'+str(int(torch.argmax(pred[0][1:])))
            str_tar = str(int(target[0][0]))+'_'+str(int(torch.argmax(target[0][1:])))
            # eng_pred = 'pred_'+traneng(pred)+'_'
            # eng_tar = 'truth_'+traneng(target)+'_'
            # rename_pred.append(str_pred)
            # rename_tar.append(str_tar)
            # eng_str = eng_pred+eng_tar+fliename[0]
            # eng_list.append(eng_str)

            
        
    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print(mAP_score)     
    # print(classification_report(target.cpu(),pred, digits=4,zero_division=1))  
    print(classification_report(torch.cat(targets).numpy(),torch.cat(predgt).numpy(), digits=4,zero_division=1))  
    print(name_list)
    print(len(name_list))
    print(rename_pred)
    print(rename_tar)
    print(rename_tar2)
    print(111)
    print(eng_list)



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
        for i, (inputData, target) in enumerate(train_loader):
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
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
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
        

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()  # (batch,3,num_classes)
        # targets_one_hot = torch.nn.functional.one_hot(target[:,1].to(torch.int64), num_classes=6)
        # target = torch.cat([target[:,0][:, None],targets_one_hot],dim=1)
    
        with torch.no_grad():
            with autocast():
                out_m = model(input.cuda())
                ema_out = ema_model.module(input.cuda())
                output_regular = Sig(out_m).cpu()
                output_ema = Sig(ema_out).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)



if __name__ == '__main__':
    main()