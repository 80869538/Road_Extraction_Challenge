import os
import torch
import torchvision
from utils import utils as utils
import glob
from matplotlib import pyplot as plt
import random


COLORS = [[0,0,0], [255,255,255]]
CLASSES = ['background', 'road']
label_values =[0,1]
RE_dir = 'data/Road_Extraction_Dataset'


def read_RE_images(RE_dir, is_train=True):
    """Read all RE feature and label images."""

    data_dir = os.path.join(RE_dir)
    JPGImages = glob.glob(data_dir + '/*.jpg')
    num_examples = len(JPGImages)
    indices = list(range(num_examples))
    random.shuffle(indices)
    train_indices = torch.tensor(indices[0:int(0.8*num_examples)])
    valid_indices = torch.tensor(indices[int(0.8*num_examples):int(0.9*num_examples)])
    print(train_indices)

    PNGImages = [JPGImage.replace('sat.jpg','mask.png') for JPGImage in JPGImages]
    
    train_features_pathes, train_labels_pathes, test_features_pathes, test_labels_pathes, valid_features_pathes, valid_labels_pathes = [], [],[],[], [],[]
    idx = 0
    for JPGPath,PNGPath in zip(JPGImages,PNGImages):
        if idx in train_indices:
            train_features_pathes.append(JPGPath)
            train_labels_pathes.append(PNGPath)
        elif idx in valid_indices:
            valid_features_pathes.append(JPGPath)
            valid_labels_pathes.append(PNGPath)
        else:
            test_features_pathes.append(JPGPath)
            test_labels_pathes.append(PNGPath)
        idx += 1
    return train_features_pathes, train_labels_pathes,test_features_pathes,test_labels_pathes, valid_features_pathes, valid_labels_pathes

#255,255,255 with class label 1 which represent road
#0,0,0 with class label 0 which represent background
def color2label():
    color2label = torch.zeros(4, dtype=torch.long)
    color2label[3] = 1
    return color2label

def label_indices(colormap, color2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = colormap/128    #threshold at 128
    idx[idx>0.5] = 1
    idx[idx<=0.5] = 0
    idx = idx.sum(axis=-1)
    return color2label[idx]

def reverse_label(colormap):
    return (colormap * 255).expand(3,1024,1024)

class RoadSegDetaset(torch.utils.data.Dataset):
    """A customized dataset to load the RE dataset."""
    def __init__(self, feature_pathes, label_pathes):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color2label = color2label()
        self.feature_pathes = feature_pathes
        self.label_pathes = label_pathes
        
        print('read ' + str(len(self.feature_pathes)) + ' examples')
    
    def normalize_image(self, img):
        return self.transform(img.float() / 255)


    def __getitem__(self, idx):
        feature = torchvision.io.read_image(self.feature_pathes[idx])
        feature = self.normalize_image(feature)                 
        label = torchvision.io.read_image(self.label_pathes[idx], mode=torchvision.io.image.ImageReadMode.RGB)
        return (feature, label_indices(label, self.color2label))
    
    def __len__(self):
        return len(self.feature_pathes)




def load_data(batch_size):
    #divide dataset into training set and testing set with ratio 9:1
    random.seed(1) 
    train_features_pathes, train_labels_pathes,test_features_pathes, test_labels_pathes, valid_features_pathes, valid_labels_pathes = read_RE_images(RE_dir, True)
    
    print(len(train_features_pathes))
    train_dataset = RoadSegDetaset(train_features_pathes,train_labels_pathes)
    valid_dataset = RoadSegDetaset(valid_features_pathes,valid_labels_pathes)
    test_dataset = RoadSegDetaset(test_features_pathes,test_labels_pathes)
    # assert len(train_dataset) + len(test_dataset) == 6226

    num_workers = utils.get_dataloader_workers()

    train_iter = torch.utils.data.DataLoader(train_dataset,
                    batch_size,shuffle=True, drop_last=True, 
                    num_workers=num_workers)
    valid_iter = torch.utils.data.DataLoader(valid_dataset ,
                    batch_size,shuffle=True, drop_last=True, 
                    num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(test_dataset,
                batch_size,shuffle=True, drop_last=True, 
                num_workers=num_workers)



    return train_iter, valid_iter ,test_iter



if __name__ == "__main__":
    train_iter,valid_iter ,test_iter = load_data(4)
    for X,Y in train_iter:
        print(X.shape)
        Y = reverse_label(Y[0])
        print(torch.unique(Y))
        