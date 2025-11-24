import json
import h5py
import os
import pickle
from PIL import Image
import io
import torch
import torchaudio
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import copy
import torch.nn.functional as F
import pandas as pd

class KineticsDataset(Dataset):

    def __init__(
        self, 
        mode,
        audio_dir,
        video_dir,
        audio_transform,
        video_transform,
        max_video_T = None,
        max_audio_T = None,
        sample_rate=24000,
        n_mels=256,
        transforms=None,
        train_path = None,
        dev_path = None,
        test_path = None
    ):
        self.data = []
        self.label = []
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_path = audio_dir
        self.visual_path = video_dir
        self.max_video_T = max_video_T
        self.max_audio_T = max_audio_T
        if mode=='train':
            # csv_path = '../nas/AdaMML/data/videos/train_split.txt'
            # csv_path = './train_split_kinetics.txt'
            csv_path = train_path
        
        elif mode=='val':
            # csv_path = '../nas/AdaMML/data/videos/val.txt'
            # csv_path = './val_split_kinetics.txt'
            csv_path = dev_path
        else:
            # csv_path = '../nas/AdaMML/data/videos/test_split.txt'
            # csv_path = './test_split_kinetics.txt'
            csv_path = test_path
            
        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0]

                if os.path.exists(self.audio_path + '/' + name + '.wav'):
                    self.data.append(name)
                    self.label.append(int(item[-1]))
        print(f"[INFO] {mode.upper()} dataset loaded with {len(self.data)} samples.") 

        self.mode = mode
        self.transforms = transforms
        self._init_atransform()

        # if "label" in self.samples_frame.columns:
        #     self.labels = torch.tensor(self.samples_frame["label"].values)
        # else:
        #     self.labels = None
    def _init_atransform(self):
        # self.aid_transform = transforms.Compose([transforms.ToTensor()])
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,  # Fast Fourier Transform 크기
            hop_length=512,  # 프레임 간 간격
            n_mels=self.n_mels  # 멜 스펙트로그램 개수
        )
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1024,  # Fast Fourier Transform 크기
            hop_length=512,  # 프레임 간 간격
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]
        # print(idx)
        # 오디오 파일 로드 (.wav)
        audio_path = os.path.join(self.audio_path, av_file + ".wav")
        waveform, sr = torchaudio.load(audio_path)  # (1, L) 형태
        # spectrogram = torch.tensor(np.expand_dims(waveform, axis=0))
        # spectrogram = torch.unsqueeze(waveform, 0)
        
        # 샘플링 레이트 변환 (필요하면 적용)
        # if sr != self.sample_rate:
        #     resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
        #     waveform = resample(waveform)

        # # 멜 스펙트로그램 변환
        # spectrogram = self.mel_spectrogram(waveform)  # (1, F, T) 형태
        # spectrogram = self.spectrogram(waveform)
        
        ### transform to spectrogram
        waveform = waveform - waveform.mean()
        norm_mean = -4.503877
        norm_std = 5.141276

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        target_length = 1024
        n_frames = fbank.shape[0]
        # print(n_frames)
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        spectrogram = (fbank - norm_mean) / (norm_std * 2)
        spectrogram = torch.unsqueeze(spectrogram, 0)

        if self.max_audio_T is not None:
            # 오디오 고정 길이 패딩
            current_T_audio = spectrogram.shape[2]
            if current_T_audio < self.max_audio_T:
                spectrogram = F.pad(spectrogram, (0, self.max_audio_T - current_T_audio))
            else:
                spectrogram = spectrogram[:, :, :self.max_audio_T]

        # Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

        if file_num == 0:
            raise RuntimeError(f"Empty video folder: {path}")

        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = max(file_num // pick_num, 1)

        # selected_frames = []
        # frame_files = os.listdir(path)
        # for i in range(pick_num):
        #     if self.mode == 'train':
        #         frame_idx = random.randint(i * seg, min((i + 1) * seg, file_num - 1))
        #     else:
        #         frame_idx = i * seg + max(seg // 2, 1)

        #     frame_path = os.path.join(path, frame_files[frame_idx])  # 프레임 파일 경로
        #     frame = Image.open(frame_path).convert('RGB')
        #     selected_frames.append(transf(frame))

        # 프레임들을 하나의 텐서로 병합 (C, T, H, W)
        # video_tensor = torch.stack(selected_frames, dim=1)
        # if self.max_video_T is not None:
        #     # 비디오 고정 길이 패딩 (T 차원)
        #     current_T_video = video_tensor.shape[1]
        #     if current_T_video < self.max_video_T:
        #         video_tensor = F.pad(video_tensor, (0, 0, 0, 0, 0, self.max_video_T - current_T_video))
        #     else:
        #         video_tensor = video_tensor[:, :self.max_video_T, :, :]
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num
        frame_files = os.listdir(path)
        for i in range(pick_num):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            frame_path = os.path.join(path, frame_files[t[i]])
            # path1.append('frame_0000' + str(t[i]) + '.jpg')
            path1.append(frame_path)
            # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            image.append(Image.open(path1[i]).convert('RGB'))

            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                video_tensor = copy.copy(image_arr[i])
            else:
                video_tensor = torch.cat((video_tensor, image_arr[i]), 1)

        label = torch.tensor(self.label[idx], dtype=torch.long)


        sample = {
            "video": video_tensor,    # (C, T, H, W)
            "audio": spectrogram,     # (1, F, T)
            "label": label,            # 정수 레이블
            "idx": idx
        }
        return sample

class KineticsDataset_sample_level(Dataset):

    def __init__(
        self, 
        mode,
        audio_dir,
        video_dir,
        audio_transform,
        video_transform,
        max_video_T = None,
        max_audio_T = None,
        contribution_list=[],
        alpha=1.0,
        func='linear',
        balance=False,
        sample_rate=16000,
        n_mels=128,
        transforms=None,
        random_state=0,
        train_path = None,
        dev_path = None,
        test_path = None
    ):
        self.data = []
        self.label = []
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_path = audio_dir
        self.visual_path = video_dir
        self.drop = []  # Modality dropping

        self.max_video_T = max_video_T
        self.max_audio_T = max_audio_T

        if mode=='train':
            csv_path = train_path
        
        elif mode=='val':
            csv_path = dev_path
        else:
            csv_path = test_path
            
        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0]

                if os.path.exists(self.audio_path + '/' + name + '.wav'):
                    self.data.append(name)
                    self.label.append(int(item[-1]))

        print(f"[INFO] {mode.upper()} dataset loaded with {len(self.data)} samples.") 

        self.mode = mode
        self.transforms = transforms    
        self._init_atransform()
        # Modality balancing
        self._balance_by_sample_level(contribution_list)
    
    def _balance_by_sample_level(self, contribution_list):

        assert len(contribution_list) == len(self.data)

        original_samples = self.data.copy()
        new_samples = []
        new_drops = []
        new_labels = []

        for idx, (contrib_audio, contrib_video) in enumerate(contribution_list):
            sample = original_samples[idx]
            label_value = self.label[idx]
            self.drop.append(0)

            if 0.4 < contrib_audio < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(1)
                    new_labels.append(label_value) 
            elif -0.1 < contrib_audio <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(1)
                    new_labels.append(label_value) 
            elif contrib_audio <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(1)
                    new_labels.append(label_value) 

            # video contribution 기준 - drop audio
            if 0.4 < contrib_video < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(2)
                    new_labels.append(label_value) 
            elif -0.1 < contrib_video <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(2)
                    new_labels.append(label_value) 
            elif contrib_video <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(2)
                    new_labels.append(label_value) 

        if new_samples:
            # self.data = pd.concat([self.data] + new_samples, ignore_index=True)
            self.data.extend(new_samples)
            self.drop.extend(new_drops)
            self.label.extend(new_labels)

        assert len(self.data) == len(self.drop), "Mismatch between samples_frame and drop after resampling."
        print(f"Sample-level resampling complete: {len(new_samples)} samples added.")
        print(f"Total samples after resampling: {len(self.data)}")

    def _init_atransform(self):
        # self.aid_transform = transforms.Compose([transforms.ToTensor()])
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,  # Fast Fourier Transform 크기
            hop_length=512,  # 프레임 간 간격
            n_mels=self.n_mels  # 멜 스펙트로그램 개수
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]
        drop_flag = self.drop[idx] if idx < len(self.drop) else 0 

        label = torch.tensor(self.label[idx], dtype=torch.long)

        # 오디오 파일 로드 (.wav)
        audio_path = os.path.join(self.audio_path, av_file + ".wav")
        waveform, sr = torchaudio.load(audio_path)  # (1, L) 형태

        waveform = waveform - waveform.mean()
        norm_mean = -4.503877
        norm_std = 5.141276

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        target_length = 1024
        n_frames = fbank.shape[0]
        # print(n_frames)
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        spectrogram = (fbank - norm_mean) / (norm_std * 2)
        spectrogram = torch.unsqueeze(spectrogram, 0)
        
        if self.max_audio_T is not None:
            # 오디오 고정 길이 패딩
            current_T_audio = spectrogram.shape[2]
            if current_T_audio < self.max_audio_T:
                spectrogram = F.pad(spectrogram, (0, self.max_audio_T - current_T_audio))
            else:
                spectrogram = spectrogram[:, :, :self.max_audio_T]

        # Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

        if file_num == 0:
            raise RuntimeError(f"Empty video folder: {path}")

        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = max(file_num // pick_num, 1)

        # selected_frames = []
        # frame_files = os.listdir(path)
        # for i in range(pick_num):
        #     if self.mode == 'train':
        #         frame_idx = random.randint(i * seg, min((i + 1) * seg, file_num - 1))
        #     else:
        #         frame_idx = i * seg + max(seg // 2, 1)

        #     frame_path = os.path.join(path, frame_files[frame_idx])  # 프레임 파일 경로
        #     frame = Image.open(frame_path).convert('RGB')
        #     selected_frames.append(transf(frame))

        # # 프레임들을 하나의 텐서로 병합 (C, T, H, W)
        # video_tensor = torch.stack(selected_frames, dim=1)

        # if self.max_video_T is not None:
        #     # 비디오 고정 길이 패딩 (T 차원)
        #     current_T_video = video_tensor.shape[1]
        #     if current_T_video < self.max_video_T:
        #         video_tensor = F.pad(video_tensor, (0, 0, 0, 0, 0, self.max_video_T - current_T_video))
        #     else:
        #         video_tensor = video_tensor[:, :self.max_video_T, :, :]
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num
        frame_files = os.listdir(path)
        for i in range(pick_num):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            frame_path = os.path.join(path, frame_files[t[i]])
            # path1.append('frame_0000' + str(t[i]) + '.jpg')
            path1.append(frame_path)
            # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            image.append(Image.open(path1[i]).convert('RGB'))

            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                video_tensor = copy.copy(image_arr[i])
            else:
                video_tensor = torch.cat((video_tensor, image_arr[i]), 1)


        sample = {
            "video": video_tensor,    # (C, T, H, W)
            "audio": spectrogram,     # (1, F, T)
            "label": label,            # 정수 레이블
            "drop" : drop_flag
        }
        return sample