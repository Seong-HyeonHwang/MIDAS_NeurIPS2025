import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import os

import json
from collections import Counter
import matplotlib.pyplot as plt


class Food101Dataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        image_transform,
        text_transform,
        phase="train",
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = pd.read_json(data_path, lines=True)
        self.dev_limit = dev_limit
        self.img_dir = img_dir

        self.samples_frame = self.samples_frame.reset_index(drop=True)
        
        # self.samples_frame.img = self.samples_frame.apply(
        #     lambda row: os.path.join(img_dir, row.img) , axis=1
        # )
        self.phase = phase     
        self.image_transform = image_transform
        self.text_transform = text_transform
       
        if "label" in self.samples_frame.columns:
            self.labels = torch.tensor(self.samples_frame["label"].values)
        else:
            self.labels = None

    def __len__(self):
        return len(self.samples_frame)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples_frame.loc[idx]
        split_folder = sample["original_split"]
        img_path = os.path.join(self.img_dir, split_folder, sample["img"])

        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        
        # text = torch.Tensor(
        #     self.text_transform.get_sentence_vector(
        #         self.samples_frame.loc[idx, "text"]
        #     )
        # ).squeeze()
        # text_data = self.samples_frame.loc[idx, "text"]
        if self.text_transform is not None:
            text_data = self.text_transform(sample["text"])
            text_input = {
                "hf_text_text_token_ids": text_data["input_ids"].squeeze(0),
                "hf_text_text_segment_ids": text_data.get("token_type_ids", torch.zeros_like(text_data["input_ids"])).squeeze(0),
                # "hf_text_text_segment_ids": text_data["token_type_ids"].squeeze(0),
                "hf_text_text_valid_length": text_data["attention_mask"].sum().item(),  # 마스크의 유효 길이
            }
            # text_input = {
            #     "input_ids": text_data["input_ids"].squeeze(0),
            #     "token_type_ids": text_data.get("token_type_ids", torch.zeros_like(text_data["input_ids"])).squeeze(0),
            #     "attention_mask": text_data["attention_mask"].squeeze(0),
            # }
        else:
            # raise ValueError("text_transform is not initialized or is None.")
            text_input = sample["text"]

        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [sample["label"]]
            ).long().squeeze()
            sample = {
                "idx": idx, 
                "image": image, 
                "text": text_input, 
                "label": label
            }
        else:
            sample = {
                "idx": idx, 
                "image": image, 
                "text": text_input
            }
        return sample

class Food101Dataset_sample_level(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        img_dir,
        image_transform,
        text_transform,
        phase="train",
        contribution_list=[],  # (text_contribution, image_contribution) 리스트
        alpha=1.0,
    ):
        self.samples_frame = pd.read_json(data_path, lines=True)
        self.img_dir = img_dir
        self.drop = []
        self.alpha = alpha

        self.samples_frame = self.samples_frame.reset_index(drop=True)
        # self.samples_frame.img = self.samples_frame.apply(
        #     lambda row: os.path.join(img_dir, row.img), axis=1
        # )
        self.phase = phase    
        self.image_transform = image_transform
        self.text_transform = text_transform

        if "label" in self.samples_frame.columns:
            self.labels = torch.tensor(self.samples_frame["label"].values)
        else:
            self.labels = None

        self._balance_by_sample_level(contribution_list)

    def _balance_by_sample_level(self, contribution_list):
        assert len(contribution_list) == len(self.samples_frame)

        original_samples = self.samples_frame.copy()
        new_samples = []
        new_drops = []

        for idx, (contrib_text, contrib_img) in enumerate(contribution_list):
            sample = original_samples.iloc[[idx]]
            self.drop.append(0)

            # if contrib_text < 1:
            #     freq_text = max(1, int(self.alpha * (1.0 - contrib_text)))
            #     for _ in range(freq_text):
            #         new_samples.append(sample)
            #         new_drops.append(1)  # drop image

            # if contrib_img < 1:
            #     freq_img = max(1, int(self.alpha * (1.0 - contrib_img)))
            #     for _ in range(freq_img):
            #         new_samples.append(sample)
            #         new_drops.append(2)  # drop text
            # Text contribution 기준 - drop image
            if 0.4 < contrib_text < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(1)
            elif -0.1 < contrib_text <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(1)
            elif contrib_text <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(1)

            # Image contribution 기준 - drop text
            if 0.4 < contrib_img < 1:
                for _ in range(1):
                    new_samples.append(sample)
                    new_drops.append(2)
            elif -0.1 < contrib_img <= 0.4:
                for _ in range(2):
                    new_samples.append(sample)
                    new_drops.append(2)
            elif contrib_img <= -0.1:
                for _ in range(3):
                    new_samples.append(sample)
                    new_drops.append(2)

        if new_samples:
            self.samples_frame = pd.concat([self.samples_frame] + new_samples, ignore_index=True)
            self.drop.extend(new_drops)

        assert len(self.samples_frame) == len(self.drop), "Mismatch between samples_frame and drop after resampling."
        print(f"Sample-level resampling complete: {len(new_samples)} samples added.")
        print(f"Total samples after resampling: {len(self.samples_frame)}")

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        drop_flag = self.drop[idx] if idx < len(self.drop) else 0

        sample = self.samples_frame.loc[idx]
        split_folder = sample["original_split"]
        img_path = os.path.join(self.img_dir, split_folder, sample["img"])

        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        text_data = self.text_transform(self.samples_frame.loc[idx, "text"])
        text_input = {
            "hf_text_text_token_ids": text_data["input_ids"].squeeze(0),
            "hf_text_text_segment_ids": text_data.get("token_type_ids", torch.zeros_like(text_data["input_ids"])).squeeze(0),
            "hf_text_text_valid_length": text_data["attention_mask"].sum().item(),
        }

        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [sample["label"]]
            ).long().squeeze()
            sample = {
                "idx": idx, 
                "image": image,
                "text": text_input,
                "label": label,
                "drop": drop_flag
            }
        else:
            sample = {
                "idx": idx, 
                "image": image,
                "text": text_input,
                "drop": drop_flag
            }

        return sample