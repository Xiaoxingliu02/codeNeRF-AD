import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
from utils import cv_utils
import csv
import torch


class AusDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AusDataset, self).__init__(opt, is_for_train)
        self._name = 'AusDataset'

        # read dataset
        #self._read_dataset_paths()
        self._root = self._opt.data_dir
        self._dataset_size=6169
        self._dataset_size1=533
        self._csv_dir = os.path.join(self._root, "crop/")
        self._imgs_dir = os.path.join(self._root, "crop/")

        # read ids
        #use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        #use_ids_filename = os.listdir(self._imgs_dir)
        #use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._ids = []#use_ids_filename[0:7271]#self._read_ids(use_ids_filepath)
        self._ids1 = []
        for i in range(0,6169):
            self._ids+=str(i)
        for i in range(6169,6702):
            self._ids1+=str(i)       

    def __getitem__(self, index):
        
        #assert (index < self._dataset_size)

        # start_time = time.time()
        if self._is_for_train is True:
            real_img = None
            real_cond = None
            while real_img is None or real_cond is None:
                # if sample randomly: overwrite index
                if not self._opt.serial_batches:
                    index = random.randint(0, self._dataset_size - 1)
    
                # get sample data
                sample_id = self._ids[index]
    
                real_img, real_img_path = self._get_img_by_id(sample_id)
                real_cond = self._get_cond_by_id(sample_id)
    
                if real_img is None:
                    print( 'error reading image %s, skipping sample' % sample_id)
                if real_cond is None:
                    print ('error reading aus %s, skipping sample' % sample_id)
    
            desired_cond = self._generate_random_cond()
    
            # transform data
            img = self._transform(Image.fromarray(real_img))
            
    
            # pack data
            sample = {'real_img': img,
                      'real_cond': real_cond,
                      'desired_cond': desired_cond,
                      'sample_id': sample_id,
                      'real_img_path': real_img_path
                      }
    
            # print (time.time() - start_time)
    
            return sample
        else:
            real_img = None
            real_cond = None
            while real_img is None or real_cond is None:
                # if sample randomly: overwrite index
                if not self._opt.serial_batches:
                    index = random.randint(0, self._dataset_size1 - 1)
    
                # get sample data
                sample_id = self._ids1[index]
    
                real_img, real_img_path = self._get_img_by_id(sample_id)
                real_cond = self._get_cond_by_id(sample_id)
    
                if real_img is None:
                    print( 'error reading image %s, skipping sample' % sample_id)
                if real_cond is None:
                    print ('error reading aus %s, skipping sample' % sample_id)
    
            desired_cond = self._generate_random_cond()
    
            # transform data
            img = self._transform(Image.fromarray(real_img))
            
    
            # pack data
            sample = {'real_img': img,
                      'real_cond': real_cond,
                      'desired_cond': desired_cond,
                      'sample_id': sample_id,
                      'real_img_path': real_img_path
                      }
    
            # print (time.time() - start_time)
    
            return sample

    def __len__(self):
    
        if self._is_for_train is True:
            return self._dataset_size
        else:
            return self._dataset_size1

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.images_folder)

        # read ids
        #use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        #use_ids_filename = os.listdir(self._imgs_dir)
        #use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._ids = []#use_ids_filename[0:7271]#self._read_ids(use_ids_filepath)

        for i in range(0,6169):
            self._ids+=str(i)+'.jpg'
            conds_filepath+=str(i)+'.csv'
        # read aus
        #conds_filepath = os.path.join(self._root, self._opt.aus_file)
        self._conds = self._read_conds(conds_filepath)

        #self._ids = list(set(self._ids).intersection(set(self._conds.keys())))

        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def _read_conds(self, file_path):
    
        #with open(file_path, 'rb') as f:
        #   return pickle.load(f,encoding='bytes')
         
        with open(os.path.join(self.au_seq[idx],str(i)+".csv")) as file:
            reader = csv.reader(file)
            for  row,index in enumerate(reader):
                if row==1:
                    input_aus.append([index[26],index[28],index[31],index[33],index[34]])

    def _get_cond_by_id(self, id):
        
        '''
        if id.encode('utf-8') in self._conds:
            return self._conds[id.encode('utf-8')]/5.0
        else:
            return None
        '''
        with open(os.path.join(self._csv_dir+id+".csv")) as file:
            reader = csv.reader(file)
            for  row,index in enumerate(reader):
                if row==1:
                    input_aus=[float(index[9]),float(index[11]),float(index[14]),float(index[16]),float(index[17])]
        return torch.tensor(input_aus)     

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id+'.jpg')
        return cv_utils.read_cv2_img(filepath), filepath

    def _generate_random_cond(self):
        cond = None
        while cond is None:
            rand_sample_id = self._ids[random.randint(0, self._dataset_size - 1)]
            cond = self._get_cond_by_id(rand_sample_id)
            cond += np.random.uniform(-0.1, 0.1, cond.shape)
        return cond
