import os
from PIL import Image
import numpy as np
import random
import torch
import cv2
import copy
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms


class OVDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.root = opt.dataroot

        self.opt = opt
        self.model = opt.model
        self.img_dir = os.path.join(opt.dataroot, 'all')
        self.parsing_dir = os.path.join(opt.dataroot, 'all_parsing')
        self.keypoints_dir = os.path.join(opt.dataroot, 'all_keypoints')
        self.densepose_dir = os.path.join(opt.dataroot, 'all_densepose')
        self.person_clothes_txt_path = os.path.join(opt.dataroot, 'all_poseA_poseB_clothes_0607.txt')
        self.clothes_category_txt_path = os.path.join(opt.dataroot, 'clothes_category_pairs.txt')

        self.img_name_list = []
        with open(self.person_clothes_txt_path, 'r') as f:
            for line in f.readlines():
                personA, personB, cloth, mode = line.strip().split()
                if mode == opt.phase:
                    key = 'half_front'
                    parsing_pathA = os.path.join(self.parsing_dir, personA.replace('.jpg', '.png'))
                    parsing_pathB = os.path.join(self.parsing_dir, personB.replace('.jpg', '.png'))
                    densepose_pathA = os.path.join(self.densepose_dir, personA.replace('.jpg', '_IUV.png'))
                    densepose_pathB = os.path.join(self.densepose_dir, personB.replace('.jpg', '_IUV.png'))
                    if key in personA and personA not in self.img_name_list:
                        if not os.path.exists(parsing_pathA) or not os.path.exists(densepose_pathA):
                            continue
                        self.img_name_list.append(personA)
                    if key in personB and personB not in self.img_name_list:
                        if not os.path.exists(parsing_pathB) or not os.path.exists(densepose_pathB):
                            continue
                        self.img_name_list.append(personB)

        img_name_list_A = self.img_name_list
        img_name_list_B = copy.deepcopy(img_name_list_A)
        random.shuffle(img_name_list_B)

        self.img_name_list_A = img_name_list_A
        self.img_name_list_B = img_name_list_B
        self.dataset_size = len(self.img_name_list)

        self.transform_affine = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomPerspective(0.3),
                transforms.RandomAffine(degrees=10, translate=(
                    0.1, 0.1), scale=(0.8, 1.2), shear=20),
                transforms.ToTensor()])  # change to [C, H, W]

        self.transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def custom_transform(self, seg):
        num_channel = seg.shape[0]
        seg[0] = 0
        affine_seg = np.zeros(shape=seg.shape, dtype=seg.dtype)
        for i in range(num_channel):
            if i == 3 or i == 5 or i == 7 or i == 10 or i == 14 or i == 15:
                affine_seg[i] = self.transform_affine(seg[i])
            else:
                affine_seg[i] = torch.from_numpy(seg[i])
            affine_seg[0] = affine_seg[0] + affine_seg[i]
        affine_seg[0] = 1 - np.clip(affine_seg[0], 0.0, 1.0)
        return affine_seg

    def __getitem__(self, index):
        # input A (label maps source)
        person_name_A = self.img_name_list_A[index]
        A_img_path = os.path.join(self.img_dir, person_name_A)
        A_img = Image.open(A_img_path)

        A_parse_path = os.path.join(self.parsing_dir, person_name_A.replace('.jpg', '.png'))
        A_seg = self.parsing_embedding(A_parse_path, 'seg')  # channel(20), H, W

        # original seg mask
        parse_A = Image.open(A_parse_path)
        parse_A = np.array(parse_A)
        parse_A = torch.tensor(parse_A, dtype=torch.long)

        # input B (label maps target)
        person_name_B = self.img_name_list_B[index]

        # densepose maps
        if 'shape' in self.model:
            person_name_B = person_name_A
            A_seg_tensor = self.custom_transform(A_seg)
        elif 'app' in self.model:
            person_name_B = person_name_A
            A_seg_tensor = torch.from_numpy(A_seg)
        else:
            A_seg_tensor = torch.from_numpy(A_seg)

        B_img_path = os.path.join(self.img_dir, person_name_B)
        B_img = Image.open(B_img_path)

        B_parse_path = os.path.join(self.parsing_dir, person_name_B.replace('.jpg', '.png'))
        B_seg = self.parsing_embedding(B_parse_path, 'seg')  # channel(20), H, W

        # original seg mask
        parse_B = Image.open(B_parse_path)
        parse_B = np.array(parse_B)
        parse_B = torch.tensor(parse_B, dtype=torch.long)

        dense_path = os.path.join(self.densepose_dir, person_name_A.replace('.jpg', '_IUV.png'))
        dense_img = cv2.imread(dense_path).astype('uint8')
        dense_img_parts_embeddings = self.parsing_embedding(
            dense_img[:, :, 0], 'densemap')

        dense_img_parts_embeddings = np.transpose(
            dense_img_parts_embeddings, axes=(1, 2, 0))
        dense_img_final = np.concatenate(
            (dense_img_parts_embeddings, dense_img[:, :, 1:]), axis=-1)  # channel(27), H, W

        B_seg_tensor = torch.from_numpy(B_seg)
        dense_img_final = torch.from_numpy(np.transpose(dense_img_final, axes=(2, 0, 1)))

        A_img_tensor = self.transform_img(A_img)
        B_img_tensor = self.transform_img(B_img)

        input_dict = {'img_A': A_img_tensor, 'seg_map_A': A_seg_tensor,  'parse_A': parse_A, 'A_path': person_name_A,
                      'dense_map': dense_img_final, 'img_B': B_img_tensor, 'parse_B': parse_B, 'seg_map_B': B_seg_tensor}

        return input_dict

    def parsing_embedding(self, parse_obj, parse_type):

        if parse_type == "seg":
            parse = Image.open(parse_obj)
            parse = np.array(parse)
            parse_channel = 20

        elif parse_type == "densemap":
            parse = np.array(parse_obj)
            parse_channel = 25

        parse_emb = []

        for i in range(parse_channel):
            parse_emb.append((parse == i).astype(np.float32).tolist())

        parse = np.array(parse_emb).astype(np.float32)
        return parse

    def __len__(self):
        return len(self.img_name_list) // self.opt.batch_size * self.opt.batch_size

    def name(self):
        return 'OvShapeDataset'

