import os
import cv2
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

class CoarseDataset(Dataset):
    def __init__(self, anno_path):
        self.root_dir = '../data/'
        self.img_dir = self.root_dir + 'cepha400/'
        self.image_lst = glob(self.root_dir + 'cepha400/*.jpg')
        self.landmarks = []
        self.img_size = 224
        
        df = pd.read_csv(anno_path)
        
        for i in range(df.shape[0]):
            sr = df.iloc[i].tolist()
            self.landmarks.append(sr)
                
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark = self.landmarks[index]
                        
        img_path = ''
        for path in self.image_lst:
            if path.split('/')[-1] == landmark[0]:
                img_path = path
                break
                
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)
        
        org_size = image.shape
        w_ratio, h_ratio = org_size[0]/self.img_size, org_size[1]/self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensor()
        ])
        data = transform(image=image)
        res = []
        for idx in range(1, len(landmark), 2):
                res.append(int(landmark[idx]/h_ratio))
                res.append(int(landmark[idx+1]/w_ratio))
                
        return data['image'], res, [img, landmark]


class FineDataset(Dataset):
    def __init__(self, num, anno_path):
        self.root_dir = 'roi/'
        self.img_dir = self.root_dir + num + '/'
        self.image_lst = os.listdir(self.img_dir)
        self.landmarks = []
        self.img_size = 224
        
        df = pd.read_csv(anno_path, header=None)  
        
        for i in range(df.shape[0]):
            sr = df.iloc[i].tolist()
            self.landmarks.append(sr)
                
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark = self.landmarks[index]
                        
        img_path = ''
        for path in self.image_lst:            
            if path == landmark[0]:
                img_path = self.img_dir + path
                
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)
        
        org_size = image.shape
        w_ratio, h_ratio = org_size[0]/self.img_size, org_size[1]/self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensor()
        ])
        data = transform(image=image)
        res = []
        for idx in range(1, len(landmark), 2):
            res.append(int(landmark[idx]/h_ratio))
            res.append(int(landmark[idx+1]/w_ratio))

        return data['image'], res, [img, landmark]