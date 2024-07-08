import os
import pickle
import torch
import numpy as np
import random
import sys
import random
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from scipy import ndimage
from skimage import data,filters,feature
from scipy.ndimage import distance_transform_edt as distance
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

#import matplotlib.pyplot as plt

# from transform1 import randonm_resize, random_rotate, rotate_resize


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file, input_size, state):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.fnames = []
        self.healthy=[]
        self.input_size=input_size
        self.state=state
        if state =='Train':
            self.istrain= True
        else:
            self.istrain= False
        

        with open(list_file) as f:
            lines = f.readlines()
        for line in lines:
            self.fnames.append(line[:-1])
            content=line[:-1]
            if content.split(' ')[1]==0:
                self.healthy.append(line[:-1])

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.

        idx = idx % self.num_samples
        fname1,fnameint = self.fnames[idx].split(' ',1)
        # cls_fname1 = int(self.fnames[idx].split(' ')[1])
        # cls_fname2 = int(self.fnames[idx].split(' ')[2])
        
        cls_fname1,cls_fname2,cls_fname3,cls_fname4,cls_fname5,cls_fname6,cls_fname7,cls_fname8,cls_fname9,cls_fname10,cls_fname11,cls_fname12,cls_fname13,cls_fname14,cls_fname15,cls_fname16,cls_fname17,cls_fname18,cls_fname19,cls_fname20 = fnameint.split(' ')
        cls_fname1 = int(cls_fname1)
        cls_fname2 = int(cls_fname2)
        cls_fname3 = int(cls_fname3)
        cls_fname4 = int(cls_fname4)
        cls_fname5 = int(cls_fname5)
        cls_fname6 = int(cls_fname6)
        cls_fname7 = int(cls_fname7)
        cls_fname8 = int(cls_fname8)
        cls_fname9 = int(cls_fname9)
        cls_fname10 = int(cls_fname10)
        cls_fname11 = int(cls_fname11)
        cls_fname12 = int(cls_fname12)
        cls_fname13 = int(cls_fname13)
        cls_fname14 = int(cls_fname14)
        cls_fname15 = int(cls_fname15)
        cls_fname16 = int(cls_fname16)
        cls_fname17 = int(cls_fname17)
        cls_fname18 = int(cls_fname18)
        cls_fname19 = int(cls_fname19)
        cls_fname20 = int(cls_fname20)

        img = Image.open(os.path.join(self.root[0], fname1)).convert('RGB')
        #######################################################################
        #boundary = self.dist_image__(mask)
        img = self.build_transform(self.istrain,img) 
        cls_fnames = torch.Tensor([cls_fname1,cls_fname2,cls_fname3,cls_fname4,cls_fname5,cls_fname6,cls_fname7,cls_fname8,cls_fname9,cls_fname10,cls_fname11,cls_fname12,cls_fname13,cls_fname14,cls_fname15,cls_fname16,cls_fname17,cls_fname18,cls_fname19,cls_fname20])
        # return img,cls_fnames
        return img,cls_fnames
    
    def build_transform(self,is_train,img):
        resize_im = self.input_size > 32
        if is_train:
            translater = transforms.RandomAffine(5, translate=(0, 0.1), scale=(0.9, 1.1), shear=0)
            angle, translations, scale, shear = translater.get_params(translater.degrees, translater.translate,
                                                                  translater.scale, translater.shear, img.size)
            image = TF.affine(img, angle, translations, scale, shear )
            if random.random() > 0.5:
                image = TF.hflip(image)
        else:
            image = img            
        t=[]
        if resize_im:#if resize_im and not args.gen_attention_maps:
            size = int(self.input_size)
            t.append(
                transforms.Resize((size,size), interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(self.input_size))
        t.append(transforms.ToTensor())
        trans =transforms.Compose(t)
        image=trans(image)
        return image

    def _img_transform(self, img):
        return np.array(img)

    def _index_transform(self, index):
        return torch.LongTensor(np.array(index).astype('float32'))#

    def __len__(self):
        return self.num_samples
    



