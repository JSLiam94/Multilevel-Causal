import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
import torchvision
import math
import numpy as np
from .classifier import Interventional_Classifier3fu2
from .vt_res32fu import VisionTransformerres32fu


import requests
from timm.models.helpers import load_pretrained
from  . import vit_seg_configs as configs

CONFIGS_ViT_seg = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}



class IDA33u(Module):
    def __init__(self,backbone="resnet101",num_classes=8,pretrain=None,use_intervention=False,heavy=False):
        super(IDA33u,self).__init__()
        if backbone=="resvit32fu":
            self.backbone = resvit32fu(num_classes,pretrain,large = True) 
        self.feat_dim = self.backbone.feat_dim
        self.use_intervention = use_intervention
        
        if not use_intervention:
            self.clf = nn.Linear(self.feat_dim,num_classes)
        else:
            self.clf = Interventional_Classifier3fu2(num_classes=num_classes, feat_dim=self.feat_dim, num_head=4, beta=0.03125, heavy=False)
    def forward(self,x):
        feats,x_logits = self.backbone(x)
       
        if self.use_intervention:
            logits = self.clf(feats,x_logits)
        else:
            logits = self.clf(feats.flatten(2).mean(-1))
        return logits


 
class resnet101_backbone(Module):
    def __init__(self, pretrain):
        super(resnet101_backbone,self).__init__()
        res101 = torchvision.models.resnet101(pretrained=True)#torchvision.models.resnet101(pretrained=True)
        if pretrain:
            path = pretrain
            state = torch.load(path, map_location='cpu')
            if type(state)==dict and "state_dict" in state:
                #res101 = nn.DataParallel(res101)
                res101.load_state_dict(state["state_dict"])
                res101 = res101.module
            else:
                res101.load_state_dict(state)
        numFit = res101.fc.in_features
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
        
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
        
        return feats

    
    

class resvit32fu(Module):
    def __init__(self,num_classes,pretrain,large=False):
        super(resvit32fu,self).__init__()
        configs=CONFIGS_ViT_seg['R50-ViT-B_16']
        configs.patches.grid =(int(512/ 16), int(512/ 16))
        self.model = VisionTransformerres32fu(configs,img_size=512, num_classes=num_classes)
        #self.model.load_from(weights=np.load('R50+ViT-B_16.npz'))
        numFit =configs.hidden_size
        self.feat_dim = numFit

    def forward(self,x):
        feats,x_logits= self.model.forward_features(x)
        return feats,x_logits
