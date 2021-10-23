import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import random
import torch.utils.data as data

SYSU_DATA_DIR = '../IVReIDData/sysu_data/'
REGDB_DATA_DIR = '../IVReIDData/RegDB/'


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (288,144)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

    
class RandomCrop(object):
    def __init__(self, image_w, image_h):
        self.image_h = image_h
        self.image_w = image_w
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']

        return {'image': image, 'thermal': thermal, 'label': label}

class RandomFlip(object):
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image   = image.transpose(Image.FLIP_TOP_BOTTOM)
            thermal = thermal.transpose(Image.FLIP_TOP_BOTTOM)
            label   = label.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': image, 'thermal': thermal, 'label': label}    


################################
# SYSU
################################

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class SYSUData3(data.Dataset):
    def __init__(self, transform=None, colorIndex = None, thermalIndex = None):
        self.train_color_image = np.load(SYSU_DATA_DIR + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(SYSU_DATA_DIR + 'train_rgb_resized_label.npy')

        self.train_thermal_image = np.load(SYSU_DATA_DIR + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(SYSU_DATA_DIR + 'train_ir_resized_label.npy')
    
        self.train_color_mask = np.load(SYSU_DATA_DIR + 'train_rgb_resized_heatmap.npy')
        self.train_thermal_mask = np.load(SYSU_DATA_DIR + 'train_ir_resized_heatmap.npy')

        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        self.transform = transform
        self.totensor = transforms.ToTensor()
    
    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_mask[self.cIndex[index]], self.train_thermal_mask[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        mask1 = self.totensor(mask1)
        mask2 = self.totensor(mask2)

        return img1, img2, target1, target2, mask1, mask2

    def __len__(self):
        return len(self.train_color_label)


class SYSUData4(data.Dataset):
    def __init__(self, transform=None, colorIndex = None, thermalIndex=None):
        self.train_color_image = np.load(SYSU_DATA_DIR + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(SYSU_DATA_DIR + 'train_rgb_resized_label.npy')

        self.train_thermal_image = np.load(SYSU_DATA_DIR + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(SYSU_DATA_DIR + 'train_ir_resized_label.npy')

        self.train_color_mask = np.load(SYSU_DATA_DIR + 'train_rgb_merged_heatmap.npy')
        self.train_thermal_mask = np.load(SYSU_DATA_DIR + 'train_ir_merged_heatmap.npy')

        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        self.transform = transform
        self.randomerasing = transforms.RandomErasing()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()


    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_mask[self.cIndex[index]], self.train_thermal_mask[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        mask1 = self.transform(mask1)
        mask2 = self.transform(mask2)
        
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
        
        img1 = self.totensor(img1)
        img2 = self.totensor(img2)
        mask1 = self.totensor(mask1)
        mask2 = self.totensor(mask2)
        
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        img1 = self.randomerasing(img1)
        img2 = self.randomerasing(img2)
        mask1 = self.randomerasing(mask1)
        mask2 = self.randomerasing(mask2)

        return img1, img2, target1, target2, mask1, mask2

    def __len__(self):
        return len(self.train_color_label)



################################
# Regdb
################################

class RegDBData(data.Dataset):
    def __init__(self, trial, transform=None, colorIndex = None, thermalIndex = None):
        train_color_list   = REGDB_DATA_DIR + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = REGDB_DATA_DIR + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(REGDB_DATA_DIR + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(REGDB_DATA_DIR+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # RGB format
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData2(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        self.data_dir = data_dir
        train_color_list  = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)
        self.train_color_image = self.load_img(color_img_file)
        self.train_thermal_image = self.load_img(thermal_img_file)

        self.train_color_image_mask = np.load('./regdb_data/train_rgb_trial{}_resized_heatmap.npy'.format(trial))
        self.train_thermal_image_mask = np.load('./regdb_data/train_ir_trial{}_resized_heatmap.npy'.format(trial)) 

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.transform = transform
        self.totensor = transforms.ToTensor()
    
    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_image_mask[self.cIndex[index]],  self.train_thermal_image_mask[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        mask1 = self.totensor(mask1)
        mask2 = self.totensor(mask2)
        
        return img1, img2, target1, target2, mask1, mask2

    def load_img(self, _img_file):
        train_image = []
        for i in range(len(_img_file)):
            img = Image.open(self.data_dir+_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image.append(pix_array)
        train_image = np.array(train_image) 
        return train_image

    def __len__(self):
        return len(self.train_color_label)


class RegDBData4(data.Dataset):
    def __init__(self, trial, transform=None, colorIndex = None, thermalIndex = None):
        train_color_list  = REGDB_DATA_DIR + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = REGDB_DATA_DIR + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)
        self.train_color_image = self.load_img(color_img_file)
        self.train_thermal_image = self.load_img(thermal_img_file)

        self.train_color_image_mask = np.load('../IVReIDData/regdb_data/train_rgb_trial{}_merged_heatmap.npy'.format(trial))
        self.train_thermal_image_mask = np.load('../IVReIDData/regdb_data/train_ir_trial{}_merged_heatmap.npy'.format(trial)) 

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.transform = transform
        self.randomerasing = transforms.RandomErasing()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
    
    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_image_mask[self.cIndex[index]],  self.train_thermal_image_mask[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        mask1 = self.transform(mask1)
        mask2 = self.transform(mask2)
        
        # if random.random() > 0.5:
        #     img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
        # if random.random() > 0.5:
        #     img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
        
        img1 = self.totensor(np.array(img1))
        img2 = self.totensor(np.array(img2))
        mask1 = self.totensor(np.array(mask1))
        mask2 = self.totensor(np.array(mask2))
        
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        img1 = self.randomerasing(img1)
        img2 = self.randomerasing(img2)
        mask1 = self.randomerasing(mask1)
        mask2 = self.randomerasing(mask2)
        
        return img1, img2, target1, target2, mask1, mask2

    def load_img(self, _img_file):
        train_image = []
        for i in range(len(_img_file)):
            img = Image.open(REGDB_DATA_DIR+_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image.append(pix_array)
        train_image = np.array(train_image) 
        return train_image

    def __len__(self):
        return len(self.train_color_label)