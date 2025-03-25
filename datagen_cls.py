import pandas as pd
import os
import torch
from torch.utils import data
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file, input_size, state):
        '''
        Args:
          root: (str) directory to images.
          list_file: (str) path to index file (CSV format).
          input_size: (int) model input size.
          state: (str) 'Train' or 'Test' to determine data augmentation.
        '''
        self.root = root
        self.input_size = input_size
        self.state = state

        # Read the CSV file using pandas
        self.data = pd.read_csv(list_file)

        # Extract image paths and labels
        self.left_images = self.data['Left-Fundus'].tolist()
        self.right_images = self.data['Right-Fundus'].tolist()
        self.labels = self.data[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype(np.float32)

        self.num_samples = len(self.data)

    def __getitem__(self, idx):
        '''Load image and corresponding label.

        Args:
          idx: (int) index.

        Returns:
          combined_img: (tensor) combined left and right eye images.
          label: (tensor) label tensor.
        '''
        # Load left and right eye images
        left_img_path = os.path.join(self.root, self.left_images[idx])
        right_img_path = os.path.join(self.root, self.right_images[idx])

        left_img = Image.open(left_img_path).convert('RGB')
        right_img = Image.open(right_img_path).convert('RGB')

        # Apply transforms
        left_img = self.build_transform(self.state == 'Train', left_img)
        right_img = self.build_transform(self.state == 'Train', right_img)

        # Combine left and right eye images horizontally
        combined_img = Image.new('RGB', (left_img.width * 2, left_img.height))
        combined_img.paste(left_img, (0, 0))
        combined_img.paste(right_img, (left_img.width, 0))

        # Convert combined image to tensor and apply additional transforms
        combined_img = self.final_transform(combined_img)

        # Get label
        label = torch.tensor(self.labels[idx])

        return combined_img, label

    def build_transform(self, is_train, img):
        '''Build and apply image transforms for individual images.

        Args:
          is_train: (bool) whether to apply data augmentation.
          img: (PIL Image) image to transform.

        Returns:
          transformed_img: (PIL Image) transformed image.
        '''
        # Crop the image to a square based on the shorter side
        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        img = img.crop((left, top, right, bottom))

        t = []
        if is_train:
            # Data augmentation for training
            t.append(transforms.RandomAffine(5, translate=(0, 0.1), scale=(0.9, 1.1), shear=0))
            if random.random() > 0.5:
                t.append(transforms.Lambda(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)))
        else:
            # No augmentation for testing
            pass

        # Resize
        t.append(transforms.Resize((self.input_size, self.input_size), interpolation=3))

        transform = transforms.Compose(t)
        transformed_img = transform(img)

        return transformed_img

    def final_transform(self, img):
        '''Apply final transforms to the combined image.

        Args:
          img: (PIL Image) combined image to transform.

        Returns:
          transformed_img: (tensor) transformed image tensor.
        '''
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transformed_img = transform(img)

        return transformed_img

    def __len__(self):
        return self.num_samples