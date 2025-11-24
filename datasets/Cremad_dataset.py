import os
import csv
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
# import pandas as pd


class CREMADDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            mode,
            visual_path,
            audio_path,
            train_csv=None,
            val_csv=None,
            test_csv=None,
            stat_csv=None,
            fps=3,
            max_audio_T = None,
    ):
        
        self.mode = mode
        self.fps = fps
        self.visual_path = visual_path
        self.audio_path = audio_path
        self.stat_csv = stat_csv
        self.max_audio_T = max_audio_T

        # 1. Load class names
        with open(self.stat_csv, encoding='UTF-8-sig') as f:
            self.classes = sorted([row[0] for row in csv.reader(f)])

        # 2. Load CSV based on mode
        csv_file = {'train': train_csv, 'val': val_csv, 'test': test_csv}[mode]
        self.data = []
        self.data2class = {}

        with open(csv_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for item in reader:
                clip_id, emotion = item[0], item[1]
                audio_file = os.path.join(audio_path, clip_id + '.wav')  
                visual_dir = os.path.join(visual_path, clip_id)

                if (emotion in self.classes and os.path.exists(audio_file)
                        and os.path.isdir(visual_dir)
                        and len(os.listdir(visual_dir)) >= fps):
                    self.data.append(clip_id)
                    self.data2class[clip_id] = emotion

        # 3. Set transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if mode == 'train' else transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id = self.data[idx]
        label = torch.tensor(self.classes.index(self.data2class[clip_id]), dtype=torch.long)

        # Audio
        audio_path = os.path.join(self.audio_path, clip_id + ".wav")
        waveform, sr = torchaudio.load(audio_path)
         ### transform to spectrogram
        waveform = waveform - waveform.mean()
        norm_mean = -4.503877
        norm_std = 5.141276

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        target_length = 1024
        n_frames = fbank.shape[0]

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
        frame_dir = os.path.join(self.visual_path, clip_id)
        frame_files = sorted(os.listdir(frame_dir))
        total_frames = len(frame_files)
        seg = max(total_frames // self.fps, 1)

        images = []
        for i in range(self.fps):
            if self.mode == 'train':
                frame_idx = random.randint(i * seg, min((i + 1) * seg - 1, total_frames - 1))
            else:
                frame_idx = i * seg + seg // 2
                frame_idx = min(frame_idx, total_frames - 1)

            frame_path = os.path.join(frame_dir, frame_files[frame_idx])
            image = Image.open(frame_path).convert('RGB')
            images.append(self.transform(image).unsqueeze(1))  # [C, 1, H, W]

        video = torch.cat(images, dim=1)  # [C, T, H, W]

        return {
            'audio': spectrogram,
            'video': video,
            'label': label,
            'idx': idx
        }

class CREMADDataset_sample_level(torch.utils.data.Dataset):
    def __init__(
            self, 
            mode,
            visual_path,
            audio_path,
            train_csv=None,
            val_csv=None,
            test_csv=None,
            stat_csv=None,
            contribution_list=[], 
            alpha=1.0,
            fps=3,
            max_audio_T = None,
    ):
        
        self.mode = mode
        self.fps = fps
        self.visual_path = visual_path
        self.audio_path = audio_path
        self.stat_csv = stat_csv
        self.max_audio_T = max_audio_T
        self.drop = []  # Modality dropping

        # 1. Load class names
        with open(self.stat_csv, encoding='UTF-8-sig') as f:
            self.classes = sorted([row[0] for row in csv.reader(f)])

        # 2. Load CSV based on mode
        csv_file = {'train': train_csv, 'val': val_csv, 'test': test_csv}[mode]
        self.data = []
        self.data2class = {}

        with open(csv_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for item in reader:
                clip_id, emotion = item[0], item[1]
                audio_file = os.path.join(audio_path, clip_id + '.wav')  
                visual_dir = os.path.join(visual_path, clip_id)

                if (emotion in self.classes and os.path.exists(audio_file)
                        and os.path.isdir(visual_dir)
                        and len(os.listdir(visual_dir)) >= fps):
                    self.data.append(clip_id)
                    self.data2class[clip_id] = emotion

        # 3. Set transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if mode == 'train' else transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._balance_by_sample_level(contribution_list)

    def _balance_by_sample_level(self, contribution_list):

        assert len(contribution_list) == len(self.data)

        original_samples = self.data.copy()
        new_samples = []
        new_drops = []

        for idx, (contrib_audio, contrib_video) in enumerate(contribution_list):
            # sample = original_samples.iloc[[idx]]
            sample = original_samples[idx]
            self.drop.append(0)

            if 0.4 < contrib_audio < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(1)
            elif -0.1 < contrib_audio <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(1)
            elif contrib_audio <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(1)

            # video contribution 기준 - drop audio
            if 0.4 < contrib_video < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(2)
            elif -0.1 < contrib_video <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(2)
            elif contrib_video <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(2)

        if new_samples:
            # self.data = pd.concat([self.data] + new_samples, ignore_index=True)
            self.data.extend(new_samples)
            self.drop.extend(new_drops)

        assert len(self.data) == len(self.drop), "Mismatch between data and drop after resampling."
        print(f"Sample-level resampling complete: {len(new_samples)} samples added.")
        print(f"Total samples after resampling: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id = self.data[idx]
        drop_flag = self.drop[idx] if idx < len(self.drop) else 0
        # print(drop_flag)

        label = torch.tensor(self.classes.index(self.data2class[clip_id]), dtype=torch.long)

        # Audio
        audio_path = os.path.join(self.audio_path, clip_id + ".wav")
        waveform, sr = torchaudio.load(audio_path)
         ### transform to spectrogram
        waveform = waveform - waveform.mean()
        norm_mean = -4.503877
        norm_std = 5.141276

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        target_length = 1024
        n_frames = fbank.shape[0]

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
        frame_dir = os.path.join(self.visual_path, clip_id)
        frame_files = sorted(os.listdir(frame_dir))
        total_frames = len(frame_files)
        seg = max(total_frames // self.fps, 1)

        images = []
        for i in range(self.fps):
            if self.mode == 'train':
                frame_idx = random.randint(i * seg, min((i + 1) * seg - 1, total_frames - 1))
            else:
                frame_idx = i * seg + seg // 2
                frame_idx = min(frame_idx, total_frames - 1)

            frame_path = os.path.join(frame_dir, frame_files[frame_idx])
            image = Image.open(frame_path).convert('RGB')
            images.append(self.transform(image).unsqueeze(1))  # [C, 1, H, W]

        video = torch.cat(images, dim=1)  # [C, T, H, W]

        return {
            'audio': spectrogram,
            'video': video,
            'label': label,
            'idx': idx,
            "drop" : drop_flag
        }