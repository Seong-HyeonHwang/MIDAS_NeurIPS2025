import csv
from genericpath import isdir
import os
import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class UCF101Dataset(Dataset):

    def __init__(
        self,
        mode,
        visual_path,
        flow_path_u,
        flow_path_v,
        stat_path=None,
        v_norm = True, 
        a_norm = True, 
        name = "UCF101",
        train_path = None,
        val_path = None,
        test_path = None,
    ):
        self.mode = mode
        self.data = []
        classes = []
        data2class = {}
        self.v_norm = v_norm
        self.a_norm = a_norm
        self.stat_path = stat_path
        self.visual_path = visual_path
        self.flow_path_u = flow_path_u
        self.flow_path_v = flow_path_v
        
        # print(mode)
        if self.mode == 'train':
            csv_file = train_path
        elif self.mode == 'val':
            csv_file = val_path
        else:
            csv_file = test_path

        with open(self.stat_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")[1]
                classes.append(item)
        # print(val_path)
        with open(csv_file) as f:
            next(f)
            for line in f:
                class_name = line.split('/')[0]
                name = line.split('/')[1].split('.')[0]
                # print(name)
                v_path = os.path.join(self.visual_path, name)
                fu_path = os.path.join(self.flow_path_u, name)
                fv_path = os.path.join(self.flow_path_v, name)

                if os.path.isdir(v_path) and os.path.isdir(fu_path) and os.path.isdir(fv_path):
                # if (
                #     os.path.isdir(v_path)
                #     and has_enough_frames(fu_path)
                #     and has_enough_frames(fv_path)
                # ):
                    # print(name)
                    self.data.append(name)
                    data2class[name] = class_name   
                    # print(self.data)
        self.classes = sorted(classes)

        self.data2class = data2class
        self.class_num = len(self.classes)
        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            datum = self.data[idx]
            rgb_dir = os.path.join(self.visual_path, datum)
            u_dir = os.path.join(self.flow_path_u, datum)
            v_dir = os.path.join(self.flow_path_v, datum)
            # crop = transforms.RandomResizedCrop(112, (1/4, 1.0), (3/4, 4/3))
            if self.mode == 'train':
                rgb_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                diff_transf = [transforms.ToTensor()]

                flow_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            else:
                rgb_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor()
                ]
                diff_transf = [transforms.ToTensor()]
                flow_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                ]

            if self.v_norm:
                rgb_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                diff_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            if self.a_norm :
                flow_transf.append(transforms.Normalize([0.1307], [0.3081]))
            rgb_transf = transforms.Compose(rgb_transf)
            diff_transf = transforms.Compose(diff_transf)
            flow_transf = transforms.Compose(flow_transf)
            # folder_path = self.visual_path + datum

            ####### RGB
            file_num = 6
            
            pick_num = 3
            seg = int(file_num/pick_num)
            image_arr = []

            for i in range(pick_num):
                if self.mode == 'train':
                    chosen_index = random.randint(i*seg + 1, i*seg + seg)
                else:
                    chosen_index = i*seg + max(int(seg/2), 1)
                # path = folder_path + '/frame_0000' + str(chosen_index) + '.jpg'
                path = os.path.join(rgb_dir, f"img_0000{str(chosen_index)}.jpg")
                # print(path)
                tranf_image = rgb_transf(Image.open(path).convert('RGB'))
                image_arr.append(tranf_image.unsqueeze(0))
            
            images = torch.cat(image_arr)

            num_u = len(os.listdir(u_dir))
            pick_num = 3
            flow_arr = []
            seg = max(1, int(num_u / pick_num))

            for i in range(pick_num):
                if self.mode == 'train':
                    chosen_index = random.randint(i*seg + 1, i*seg + seg)
                else:
                    chosen_index = i*seg + max(int(seg/2), 1)
                chosen_index = min(chosen_index, num_u - 1)

                # flow_u = self.flow_path_u + datum + '/frame00' + str(chosen_index).zfill(4) + '.jpg'
                # flow_v = self.flow_path_v + datum + '/frame00' + str(chosen_index).zfill(4) + '.jpg'
                flow_u = os.path.join(u_dir, f"frame{str(chosen_index).zfill(5)}.jpg")
                flow_v = os.path.join(v_dir, f"frame{str(chosen_index).zfill(5)}.jpg")
                if not os.path.exists(flow_u) or not os.path.exists(flow_v):
                    raise FileNotFoundError(f"Missing optical flow file: {flow_u} or {flow_v}")
                u = flow_transf(Image.open(flow_u))
                v = flow_transf(Image.open(flow_v))
                # print("flow_u shape:", u.shape)
                flow = torch.cat((u,v),0)
                flow_arr.append(flow.unsqueeze(0))
                
                flow_n = torch.cat(flow_arr)
            flow_n = flow_n.permute(1, 0, 2, 3)
            images = images.permute(1, 0, 2, 3)
            sample = {
                'flow':flow_n,
                'visual':images,
                'label': self.classes.index(self.data2class[datum]),
                'raw':datum,
                'idx':idx
            }
            return sample
        except Exception as e:
            print(f"[SKIP] Sample '{self.data[idx]}' skipped due to: {e}")
            return self.__getitem__((idx + 1) % len(self))


class UCF101Dataset_sample_level(torch.utils.data.Dataset):
    def __init__(
        self,
        mode,
        visual_path,
        flow_path_u,
        flow_path_v,
        stat_path=None,
        v_norm = True, 
        a_norm = True, 
        name = "UCF101",
        train_path = None,
        val_path = None,
        test_path = None,
        contribution_list=[], 
        alpha=1.0, # alpha is not used in the balancing logic, similar to CREMAD example
    ):
        self.mode = mode
        self.data = []
        classes = []
        data2class = {}
        self.v_norm = v_norm
        self.a_norm = a_norm
        self.stat_path = stat_path
        self.visual_path = visual_path
        self.flow_path_u = flow_path_u
        self.flow_path_v = flow_path_v
        self.drop = [] # For modality dropping flags
        
        if self.mode == 'train':
            csv_file = train_path
        elif self.mode == 'val':
            csv_file = val_path
        else:
            csv_file = test_path

        with open(self.stat_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")[1]
                classes.append(item)
        
        with open(csv_file) as f:
            next(f) # skip header
            for line in f:
                class_name = line.split('/')[0]
                video_name = line.split('/')[1].split('.')[0]
                
                v_path = os.path.join(self.visual_path, video_name)
                fu_path = os.path.join(self.flow_path_u, video_name)
                fv_path = os.path.join(self.flow_path_v, video_name)

                # Basic check for directory existence
                if os.path.isdir(v_path) and os.path.isdir(fu_path) and os.path.isdir(fv_path):
                    # A more robust check would ensure enough frames exist,
                    # but following the original UCF101Dataset structure
                    self.data.append(video_name)
                    data2class[video_name] = class_name   
                    
        self.classes = sorted(classes)
        self.data2class = data2class
        self.class_num = len(self.classes)
        
        # Apply sample-level balancing/augmentation if in train mode and contribution_list is provided
        if self.mode == 'train' and contribution_list:
            self._balance_by_sample_level(contribution_list)
        else:
            # For val/test or if no contribution_list, all samples are original (drop_flag=0)
            self.drop = [0] * len(self.data)

        print(f'[{self.mode} mode] # of files = {len(self.data)} (after sample-level processing if any)')


    def _balance_by_sample_level(self, contribution_list):
        """
        Resamples data based on contribution scores.
        contribution_list: list of tuples (contrib_flow, contrib_rgb)
        """
        if not contribution_list:
            self.drop = [0] * len(self.data)
            print("Contribution list is empty. No sample-level balancing applied.")
            return

        if len(contribution_list) != len(self.data):
            # Fallback: if contribution list mismatch, treat all as original
            print(f"Warning: Mismatch between contribution_list length ({len(contribution_list)}) and data length ({len(self.data)}). No sample-level balancing applied.")
            self.drop = [0] * len(self.data)
            return

        original_samples = self.data.copy() # Keep a copy of the original data identifiers
        original_data2class = self.data2class.copy()

        new_data_samples = []
        new_drop_flags = []

        # Iterate over original samples and their contributions
        for idx, (contrib_flow, contrib_rgb) in enumerate(contribution_list):
            sample_id = original_samples[idx]
            
            # Add the original sample first
            new_data_samples.append(sample_id)
            new_drop_flags.append(0) # 0: use both modalities

            # Augment based on flow contribution (contrib_flow is analogous to contrib_audio)
            # If flow contribution is low, add samples where RGB is emphasized (drop_flag = 1, meaning drop RGB counterpart -> drop flow)
            # Following CREMA-D logic: low audio_contrib -> drop=1 (drop video).
            # So, low flow_contrib -> drop=1 (means drop RGB, i.e., focus on Flow)
            if 0.4 < contrib_flow < 1: # Moderately low flow contribution
                for _ in range(1):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(1) # Drop RGB / Emphasize Flow
            elif -0.1 < contrib_flow <= 0.4: # Low flow contribution
                for _ in range(2):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(1)
            elif contrib_flow <= -0.1: # Very low flow contribution
                for _ in range(3):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(1)

            # Augment based on RGB contribution (contrib_rgb is analogous to contrib_video)
            # If RGB contribution is low, add samples where Flow is emphasized (drop_flag = 2, meaning drop flow counterpart -> drop RGB)
            # Following CREMA-D logic: low video_contrib -> drop=2 (drop audio).
            # So, low rgb_contrib -> drop=2 (means drop Flow, i.e., focus on RGB)
            if 0.4 < contrib_rgb < 1: # Moderately low RGB contribution
                for _ in range(1):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(2) # Drop Flow / Emphasize RGB
            elif -0.1 < contrib_rgb <= 0.4: # Low RGB contribution
                for _ in range(2):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(2)
            elif contrib_rgb <= -0.1: # Very low RGB contribution
                for _ in range(3):
                    new_data_samples.append(sample_id)
                    new_drop_flags.append(2)
        
        self.data = new_data_samples
        self.drop = new_drop_flags
        # self.data2class remains the same as it maps sample_id to class, and sample_ids are reused.

        assert len(self.data) == len(self.drop), "Mismatch between data and drop flags after resampling."
        print(f"Sample-level resampling complete: {len(self.data) - len(original_samples)} samples added based on contributions.")
        print(f"Total samples after resampling: {len(self.data)}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            datum = self.data[idx] # video_name
            drop_flag = self.drop[idx] if idx < len(self.drop) else 0

            rgb_dir = os.path.join(self.visual_path, datum)
            u_dir = os.path.join(self.flow_path_u, datum)
            v_dir = os.path.join(self.flow_path_v, datum)

            if self.mode == 'train':
                rgb_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                diff_transf = [transforms.ToTensor()]

                flow_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            else: # val or test
                rgb_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor()
                ]
                diff_transf = [transforms.ToTensor()]
                flow_transf = [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                ]

            if self.v_norm: # visual (RGB) normalization
                rgb_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                diff_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            if self.a_norm : # "audio" (Flow) normalization
                flow_transf.append(transforms.Normalize([0.1307], [0.3081])) # Assuming flow data is single channel like

            rgb_transf = transforms.Compose(rgb_transf)
            diff_transf = transforms.Compose(diff_transf)
            flow_transf = transforms.Compose(flow_transf)
            
            ####### RGB
            # In UCF101, typically a fixed number of frames (e.g., 6 as per original code) are sampled
            # For RGB, 6 frames exist (img_00001.jpg to img_00006.jpg)
            file_num = 6
            
            pick_num = 3
            seg = int(file_num/pick_num)
            image_arr = []

            for i in range(pick_num):
                if self.mode == 'train':
                    # Frames are 1-indexed
                    chosen_index = random.randint(i*seg + 1, i*seg + seg)
                else:
                    chosen_index = i*seg + max(int(seg/2), 1)
                
                path = os.path.join(rgb_dir, f"img_0000{str(chosen_index)}.jpg")
                # print(path)
                tranf_image = rgb_transf(Image.open(path).convert('RGB'))
                image_arr.append(tranf_image.unsqueeze(0))
            
            images = torch.cat(image_arr) # Shape: [pick_num_rgb, C, H, W]

            ####### Flow
            num_u = len(os.listdir(u_dir))
            pick_num = 3
            flow_arr = []
            seg = max(1, int(num_u / pick_num))

            for i in range(pick_num):
                if self.mode == 'train':
                    chosen_index = random.randint(i*seg + 1, i*seg + seg)
                else:
                    chosen_index = i*seg + max(int(seg/2), 1)
                chosen_index = min(chosen_index, num_u - 1)
    
                
                flow_u = os.path.join(u_dir, f"frame{str(chosen_index).zfill(5)}.jpg")
                flow_v = os.path.join(v_dir, f"frame{str(chosen_index).zfill(5)}.jpg")
                if not os.path.exists(flow_u) or not os.path.exists(flow_v):
                    raise FileNotFoundError(f"Missing optical flow file: {flow_u} or {flow_v}")
                u = flow_transf(Image.open(flow_u))
                v = flow_transf(Image.open(flow_v))
                # print("flow_u shape:", u.shape)
                flow = torch.cat((u,v),0)
                flow_arr.append(flow.unsqueeze(0))
                
                flow_n = torch.cat(flow_arr)
                flow_arr.append(flow.unsqueeze(0))
            
                flow_n = torch.cat(flow_arr) # Shape: [pick_num_flow, 2, H, W]
            flow_n = flow_n.permute(1, 0, 2, 3)
            images = images.permute(1, 0, 2, 3)

            sample = {
                'flow': flow_n,
                'visual': images,
                'label': self.classes.index(self.data2class[datum]),
                'raw': datum, # original video name
                'idx': idx,      # index in the current (potentially resampled) dataset
                'drop': drop_flag
            }
            return sample

        except Exception as e:
            print(f"[SKIP] Sample '{self.data[idx]}' skipped due to: {e}")
            return self.__getitem__((idx + 1) % len(self))