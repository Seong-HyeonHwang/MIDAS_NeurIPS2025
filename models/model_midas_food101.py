import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from torchmetrics import functional as FM
from torchmetrics import Accuracy
from datasets.Food101_dataset import Food101Dataset
import warnings
import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from .huggingface_text import HFAutoModelForTextPrediction
from transformers import AutoTokenizer
from transformers import BertModel, BertTokenizer
import os
import sys
import re


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

class ResNet18Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.num_features = 512

        def forward(self, x):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            return x

class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_module,
        vision_module,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        use_augmentation,
        train_dataset,
        batch_size = 32,
        precompute_features=False,
        device_idx=None,
        drop=None,
        warmup_epochs=2
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.reduce_vision_dim = torch.nn.Linear(
            in_features=512,
            out_features=vision_feature_dim
        )
        self.train_dataset = train_dataset

        total_feature_dim = language_feature_dim + vision_feature_dim
        self.task_net = torch.nn.Sequential(
            torch.nn.Linear(total_feature_dim, num_classes)
        )
        self.text_classifier = torch.nn.Sequential(
            torch.nn.Linear(language_feature_dim, num_classes)
        )
        self.image_classifier = torch.nn.Sequential(
            torch.nn.Linear(vision_feature_dim, num_classes)
        )
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.loss_fn_none = torch.nn.CrossEntropyLoss(reduction='none')
        self.use_augmentation = use_augmentation
        self.warmup_epochs = warmup_epochs
        self.alpha = 1.0

    def forward(self, text, image, label=None, use_augmentation=False, current_epoch=None, is_last_batch=False):
        text_outputs = self.language_module(text)
        text_features = text_outputs["hf_text"]["features"]
        image_features = self.reduce_vision_dim(self.vision_module(image))
        
        fusion_features = torch.cat([text_features, image_features], dim=1)

        logits = self.task_net(fusion_features)
        loss = (
            self.loss_fn(logits, label)
            if label is not None else label
        )

        if use_augmentation:
            text_logits = self.text_classifier(text_features)
            image_logits = self.image_classifier(image_features)
                        
            text_probs = F.softmax(text_logits.detach(), dim=1)
            image_probs = F.softmax(image_logits.detach(), dim=1)
            text_conf = torch.gather(text_probs, 1, label.unsqueeze(1)).squeeze()
            image_conf = torch.gather(image_probs, 1, label.unsqueeze(1)).squeeze()

            uni_loss_1 = self.loss_fn(text_logits, label)
            uni_loss_2 = self.loss_fn(image_logits, label)
            uni_loss = (uni_loss_1 + uni_loss_2)/2
            
            if current_epoch < self.warmup_epochs:
                loss = uni_loss
            else:
                B = len(label)
                self.device = label.device
                random_idx = torch.randperm(B).to(self.device)
                mixed_loss = 0
                uni_ratio = image_conf.mean() / text_conf.mean()
                
                lambd_1, lambd_2 = 1, 0
                mixed_features = torch.cat([lambd_1*text_features + (1-lambd_1)*text_features[random_idx], lambd_2*image_features + (1-lambd_2)*image_features[random_idx]], dim=1)
                mixed_logits = self.task_net(mixed_features)
                text_weight = text_conf / (text_conf + image_conf[random_idx])
                image_weight = 1 - text_weight
                mixed_text_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label.unsqueeze(1)).squeeze()
                mixed_image_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label[random_idx].unsqueeze(1)).squeeze()
                mm_ratio_1 = mixed_image_prob.detach().mean() / mixed_text_prob.detach().mean()
                if uni_ratio > 1:
                    weighted_mixed_loss = self.alpha * text_weight * self.loss_fn_none(mixed_logits, label) + image_weight * self.loss_fn_none(mixed_logits, label[random_idx])
                else:
                    weighted_mixed_loss = text_weight * self.loss_fn_none(mixed_logits, label) + self.alpha * image_weight * self.loss_fn_none(mixed_logits, label[random_idx])
                image_sim = torch.cosine_similarity(image_features, image_features[random_idx], dim=1).detach()
                mixed_loss += (1 + (image_sim + 1)/2) * weighted_mixed_loss

                lambd_1, lambd_2 = 0, 1
                mixed_features = torch.cat([lambd_1*text_features + (1-lambd_1)*text_features[random_idx], lambd_2*image_features + (1-lambd_2)*image_features[random_idx]], dim=1)
                mixed_logits = self.task_net(mixed_features)
                text_weight = text_conf[random_idx] / (text_conf[random_idx] + image_conf)
                image_weight = 1 - text_weight
                mixed_text_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label[random_idx].unsqueeze(1)).squeeze()
                mixed_image_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label.unsqueeze(1)).squeeze()
                mm_ratio_2 = mixed_image_prob.detach().mean() / mixed_text_prob.detach().mean()
                if uni_ratio > 1:
                    weighted_mixed_loss = (self.alpha * text_weight * self.loss_fn_none(mixed_logits, label[random_idx]) + image_weight * self.loss_fn_none(mixed_logits, label))
                else:
                    weighted_mixed_loss = (text_weight * self.loss_fn_none(mixed_logits, label[random_idx]) + self.alpha * image_weight * self.loss_fn_none(mixed_logits, label))
                text_sim = torch.cosine_similarity(text_features, text_features[random_idx], dim=1).detach()
                mixed_loss += (1 + (text_sim + 1)/2) * weighted_mixed_loss

                mixed_loss = mixed_loss.mean()/2

                mm_ratio = (mm_ratio_1 + mm_ratio_2)/2
                if uni_ratio > 1:
                    if mm_ratio > uni_ratio:
                        self.alpha += 5e-2
                    else:
                        self.alpha -= 5e-2
                else:
                    if mm_ratio < uni_ratio:
                        self.alpha += 5e-2
                    else:
                        self.alpha -= 5e-2
                self.alpha = max(1, self.alpha)

                loss += 1 * mixed_loss + uni_loss
            return logits, loss, self.alpha
        return (logits, loss)

class MidasModel(pl.LightningModule):
    def __init__(self, params):
        for data_key in ["train_path", "dev_path", "img_dir",]:
            if data_key not in params.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )
        
        super(MidasModel, self).__init__()
        self.model_params = params
        self.dataset = self.model_params.get("dataset", "food101")
        self.language_feature_dim = self.model_params.get("language_feature_dim", 256)
        self.vision_feature_dim = self.model_params.get("vision_feature_dim", 300)
        self.output_path = Path(self.model_params.get("output_path", "model-outputs"))
        self.output_path.mkdir(exist_ok=True)

        self.text_transform = self._build_text_transform()
        self.train_image_transform = self._build_image_transform(phase="train")
        self.val_image_transform = self._build_image_transform(phase="val")

        self.train_dataset = self._build_dataset("train_path", phase="train")
        self.dev_dataset = self._build_dataset("dev_path", phase="val")

        self.automatic_optimization = False
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []

        self.warmup_epochs = self.model_params.get("warmup_epochs", 2)
        self.val_origin_acc = []
        self.val_swapped_acc = []
        self.val_origin_acc = []
        self.val_swapped_acc = []
        self.val_swapped_conf_1 = []
        self.val_swapped_conf_2 = []
        self.alpha_per_epoch = []
        self.alpha = 1.0

        # augmentation
        self.use_augmentation = self.model_params.get("use_augmentation", False)

    @property
    def hparams(self):
        return self.model_params
    ## Required LightningModule Methods (when validating) ##
    
    def forward(self, text, image, label=None, use_augmentation=False, current_epoch=None, is_last_batch=False):
        return self.model(text, image, label, use_augmentation, current_epoch, is_last_batch)
    
    def on_train_epoch_start(self):
        self.train_epoch_loss = 0.0

    def training_step(self, batch, batch_nb):
        self.train()
        accumulate_grad_batches = self.hparams.get("accumulate_grad_batches", 1)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        if self.use_augmentation:
            _, loss, alpha = self.forward(
                text=batch["text"], 
                image=batch["image"], 
                label=batch["label"],
                use_augmentation=True,
                current_epoch=self.current_epoch,
                is_last_batch=self.trainer.is_last_batch
            )
            self.alpha = alpha
        else:
            _, loss = self.forward(
                text=batch["text"], 
                image=batch["image"], 
                label=batch["label"],
            )
        loss = loss / accumulate_grad_batches

        self.manual_backward(loss)

        if (batch_nb + 1) % accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()
        if self.trainer.is_last_batch:
            scheduler.step()
        self.log("train_loss", loss, prog_bar=True)

        self.train_epoch_loss =  self.train_epoch_loss + loss.item()
        return {"train_loss": loss}
    
    def on_train_epoch_end(self):
        avg_train_loss = self.train_epoch_loss / len(self.train_dataloader())
        self.train_losses_per_epoch.append(avg_train_loss)
        self.train_epoch_loss = 0.0 
        self.log("avg_train_loss", avg_train_loss)

        self.alpha_per_epoch.append(self.alpha)
        ### Misaligned pair prediction accuracy ###
        if self.hparams.get("recording"):
            correct_original = 0
            correct_swapped = 0
            total_samples = 0
            conf_1_swapped = 0
            conf_2_swapped = 0
            with torch.no_grad():
                self.eval()
                for batch in self.trainer.val_dataloaders:
                    image, text, labels = batch["image"], batch["text"], batch["label"]
                    batch_size = image.size(0)
                    image = image.to(self.device)
                    # text = text.to(self.device)
                    text = {text_key: text[text_key].to(self.device) for text_key in text}
                    labels = labels.to(self.device)
                    logits, _ = self.forward(text, image, labels, use_augmentation=False)
                    _, pred = F.softmax(logits, dim=1).max(dim=1)
                    correct_original += (pred == labels).sum().item()

                    random_idx = torch.randperm(batch_size)
                    # random_text = {text_key: text[text_key][random_idx] for text_key in text}
                    random_image = image[random_idx]
                    logits, _ = self.forward(text, random_image, labels, use_augmentation=False)
                    probs = F.softmax(logits, dim=1)
                    top2_values, top2_indices = probs.topk(2, dim=1)

                    for i in range(batch_size):
                        if labels[i] == top2_indices[i, 0] and labels[random_idx][i] == top2_indices[i, 1]:
                            correct_swapped += 1
                        elif labels[i] == top2_indices[i, 1] and labels[random_idx][i] == top2_indices[i, 0]:
                            correct_swapped += 1
                    total_samples += batch_size
                    conf_1_swapped += torch.sum(torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()).item()
                    conf_2_swapped += torch.sum(torch.gather(probs, 1, labels[random_idx].unsqueeze(1)).squeeze()).item()
                print("Validation Performance:")
                print(f"  Original Fusion Features Accuracy: {correct_original / total_samples * 100:.2f}%")
                print(f"  Swapped Image Features Accuracy: {correct_swapped / total_samples * 100:.2f}%")
                print(f" swapped conf 1: {conf_1_swapped / total_samples:.2f}")
                print(f" swapped conf 2: {conf_2_swapped / total_samples:.2f}")
                self.val_origin_acc.append(correct_original / total_samples)
                self.val_swapped_acc.append(correct_swapped / total_samples)
                self.val_swapped_conf_1.append(conf_1_swapped / total_samples)
                self.val_swapped_conf_2.append(conf_2_swapped / total_samples)

    def on_validation_epoch_start(self):
        self.val_epoch_loss = 0.0

    def validation_step(self, batch, batch_nb):
        self.eval()
        logits, loss = self.forward(
            text=batch["text"], 
            image=batch["image"], 
            label=batch["label"],
        )
        
        # Get actual predictions by taking argmax
        preds = torch.argmax(logits, dim=1)
        
        acc = FM.accuracy(preds, batch["label"], task="multiclass", num_classes=self.hparams.get("num_classes", 101))
        f1 = FM.f1_score(preds, batch["label"], task="multiclass", num_classes=self.hparams.get("num_classes", 101), average="macro")
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        self.val_epoch_loss = self.val_epoch_loss + loss.item()
        return {"batch_val_loss": loss, "batch_val_acc": acc, "batch_val_f1": f1}

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_epoch_loss / len(self.val_dataloader())
        self.val_losses_per_epoch.append(avg_val_loss)
        self.val_epoch_loss = 0.0
        self.log("avg_val_loss", avg_val_loss)
        return {
            "val_loss": avg_val_loss
        }
    def configure_optimizers(self):
        # Separate parameters for main network and augnet
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.001))
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.get("lr", 0.001), momentum=0.9, weight_decay=self.hparams.get("weight_decay", 1e-4))
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
        #     # 'monitor': 'val_acc',  
        #     'monitor': 'avg_val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return [optimizer], [scheduler]
    
    # @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )

    # @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )
    
    ## Conienience Methods ##
    
    def fit(self):
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)
        
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def test_step(self, batch):
        self.eval()
        text, image, labels = batch["text"], batch["image"], batch["label"]
        preds, _ = self.forward(text, image)
        preds = torch.argmax(preds, dim=1)

       
        acc = FM.accuracy(preds, labels, task="multiclass", num_classes=self.hparams.get("num_classes", 101))
        f1 = FM.f1_score(preds, labels, task="multiclass", num_classes=self.hparams.get("num_classes", 101), average="macro")

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        return {"test_acc": acc, "test_f1": f1}
    
    def on_test_epoch_end(self):
        # Access test_acc metrics directly through self.trainer.callback_metrics
        test_acc = self.trainer.callback_metrics["test_acc"]
        test_f1 = self.trainer.callback_metrics["test_f1"]

        self.log("test_acc", test_acc)
        self.log("test_f1", test_f1)

        return {"test_acc": test_acc, "test_f1": test_f1}

    def test(self, dataloader):
        self.eval()  
        all_preds = []
        all_labels = []

        with torch.no_grad():  
            for batch in dataloader:
                output = self.test_step(batch)
                all_preds.append(output["preds"])
                all_labels.append(output["labels"])

        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        
        test_acc = FM.accuracy(all_preds, all_labels, task="multiclass", num_classes=self.hparams.get("num_classes", 101))
        test_f1 = FM.f1_score(all_preds, all_labels, task="multiclass", num_classes=self.hparams.get("num_classes", 101), average="macro")

        print(f"Test Accuracy: {test_acc.item():.4f}")
        print(f"Test F1 Score: {test_f1.item():.4f}")

    def clean_text(self,raw_text):
        t = re.sub(r'^RT[\s]+', '', raw_text)# remove old style retweet text "RT"
        t = re.sub(r'https?:\/\/.*[\r\n]*', '', t)# remove hyperlinks
        t = re.sub(r'#', '', t) # remove hashtags
        return t

    def _build_text_transform(self):
        tokenizer = AutoTokenizer.from_pretrained(
        "google/electra-small-discriminator"
        )

        def tokenize_function(text):
            text = self.clean_text(text)
            tokenized_output = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return(tokenized_output)

        print("Tokenize function initialized")
        return tokenize_function
        
    def _build_image_transform(self, phase="train"):
        resize_dim = 224
        if phase == "train":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((resize_dim, resize_dim)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:  # val or test
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((resize_dim, resize_dim)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        return transform
    
    def _build_dataset(self, dataset_key, phase="train"):
        return Food101Dataset(
            data_path=self.hparams.get(dataset_key, dataset_key),
            img_dir=self.hparams.get("img_dir"),
            image_transform=self.train_image_transform if phase == "train" else self.val_image_transform,
            text_transform=self.text_transform,
            phase=phase
            # limit training samples only
            # dev_limit=(
            #     self.hparams.get("dev_limit", None) 
            #     if "train" in str(dataset_key) else None
            # ),
            # balance=True if "train" in str(dataset_key) else False,
        )
    
    def _build_model(self):
        # language_module = BERTTextEncoder(output_dim=self.language_feature_dim)
        language_module = HFAutoModelForTextPrediction(
            prefix="hf_text",
            checkpoint_name="google/electra-small-discriminator",  
            num_classes=101
        )
        # vision_module = TimmAutoModelForImagePrediction(
        # prefix="timm_image",
        # checkpoint_name="convnext_base_in22ft1k",  
        # num_classes=101,  
        # pretrained=True,
        # ).model

        # vision_module.fc = torch.nn.Linear(
        #     in_features=vision_module.num_features,
        #     out_features=self.vision_feature_dim,
        # )
        resnet18 = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]  # remove last fc
        resnet_backbone = nn.Sequential(*modules)

        vision_module = ResNet18Wrapper(resnet_backbone)

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 101),
            loss_fn=torch.nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size", 512
            ),
            dropout_p=self.hparams.get("dropout_p", 0.5),
            use_augmentation=self.hparams.get("use_augmentation", True),
            train_dataset=self.train_dataset,
            batch_size=self.hparams.get("batch_size", 32),
            precompute_features=self.hparams.get("use_augmentation", False),
            device_idx=self.hparams.get("devices", [1]),
            drop=None,
            warmup_epochs=self.hparams.get("warmup_epochs", 2)
        )
    
    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            filename='{epoch}-{val_loss:.3f}-{val_acc:.3f}',
            monitor='val_acc',  # Metric to monitor
            mode='max',         # Save model when metric is maximized
            save_top_k=1,      # Save only the best model
            verbose=True,
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get(
                "early_stop_monitor", "val_acc"
            ),
            min_delta=self.hparams.get(
                "early_stop_min_delta", 0.001
            ),
            patience=self.hparams.get(
                "early_stop_patience", 5
            ),
            verbose=self.hparams.get("verbose", True),
            mode='max',  # Stop training when metric stops increasing
        )

        trainer_params = {
            "accelerator": self.hparams.get("accelerator", "gpu"),
            "callbacks": [checkpoint_callback, early_stop_callback],
            # "accumulate_grad_batches": self.hparams.get(
            #     "accumulate_grad_batches", 1
            # ),
            # "devices": self.hparams.get("n_gpu", 1),
            "devices": self.hparams.get("devices", [1]),
            "max_epochs": self.hparams.get("max_epochs", 100),
            # "strategy": self.hparams.get("strategy", "ddp_find_unused_parameters_true"),
            # "gradient_clip_val": self.hparams.get(
            #     "gradient_clip_value", 3
            # ),
            "num_sanity_val_steps": 0,
        }
        return trainer_params
    
    def plot_learning_curves(self):
        plt.figure(figsize=(12, 5))

        # method = "modal_swap_constraint"
        method = self.hparams.get("method", "modal_swap_constraint")
        if self.dataset == "hateful_memes":
            filename = "avg_modalities_contributions_hateful_memes_" + method + ".json"
        if self.dataset == "food101":
            filename = "avg_modalities_contributions_food101_" + method + ".json"
        contributions = [
            {
                "epoch": i,
                "text_contribution": self.cont_per_epoch[i],
                "image_contribution": self.coni_per_epoch[i]
            }
            for i in range(len(self.cont_per_epoch))
        ]

        with open(filename, "w") as f:
            json.dump(contributions, f, indent=4)

        print(f"All modality contributions saved to {filename}")

        # Loss plot
        plt.subplot(121)
        plt.plot(range(1, len(self.train_losses_per_epoch) + 1), self.train_losses_per_epoch, label="Training Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')

        plt.subplot(122)
        plt.plot(range(1, len(self.val_losses_per_epoch) + 1), self.val_losses_per_epoch, label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Validation Loss')

        plt.suptitle("Learning Curves")
        plt.tight_layout()
        if self.dataset == "hateful_memes":
            plt.savefig('learning_curve_hateful_memes_' + method + '.png')
        if self.dataset == "food101":
            plt.savefig('learning_curve_food101_' + method + ".png")

        # valuation plot
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(range(1, len(self.cont_per_epoch) + 1), self.cont_per_epoch, label="Text Contribution", marker = 'o')
        plt.plot(range(1, len(self.coni_per_epoch) + 1), self.coni_per_epoch, label="Image Contribution", marker = 'o')
        plt.xlabel('Epochs')
        plt.ylabel('Contribution')
        plt.legend()
        plt.title('Modality Contribution')

        plt.subplot(122)
        con_sum = [self.cont_per_epoch[i] + self.coni_per_epoch[i] for i in range(len(self.cont_per_epoch))]
        cont_per_epoch_norm = [cont / con_sum[i] for i, cont in enumerate(self.cont_per_epoch)]
        coni_per_epoch_norm = [coni / con_sum[i] for i, coni in enumerate(self.coni_per_epoch)]
        plt.plot(range(1, len(cont_per_epoch_norm) + 1), cont_per_epoch_norm, label="Text Contribution", marker = 'o')
        # plt.plot(range(1, len(coni_per_epoch_norm) + 1), coni_per_epoch_norm, label="Image Contribution", marker = 'o')
        plt.xlabel('Epochs')
        plt.ylabel('Contribution ratio')
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.title('Modality Contribution Ratio')

        plt.suptitle("Modality Contribution")
        plt.tight_layout()
        if self.dataset == "hateful_memes":
            plt.savefig('modality_contribution_hateful_memes_' + method + '.png')
        if self.dataset == "food101":
            plt.savefig('modality_contribution_food101_' + method + ".png")

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.val_origin_acc) + 1), self.val_origin_acc, label="Original Accuracy", marker = 'o')
        plt.plot(range(1, len(self.val_swapped_acc) + 1), self.val_swapped_acc, label="Swapped Accuracy", marker = 'o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        if self.dataset == "hateful_memes":
            plt.savefig('mmix_analysis/validation_accuracy_hateful_memes_' + method + '.png')
        if self.dataset == "food101":
            plt.savefig('mmix_analysis/validation_accuracy_food101_' + method + ".png")
        
    def save_records(self, seed):
        # Save the model
        method = self.hparams.get("method", " ")

        filename = "save_records/food101_record_" + method + "_" + str(seed) + ".json"
        contributions = [
            {
                "epoch": i,
                "original_acc": self.val_origin_acc[i],
                "misaligned_acc": self.val_swapped_acc[i],
                "alpha": self.alpha_per_epoch[i],
                "swapped_conf_1": self.val_swapped_conf_1[i],
                "swapped_conf_2": self.val_swapped_conf_2[i],
            }
            for i in range(len(self.alpha_per_epoch))
        ]

        with open(filename, "w") as f:
            json.dump(contributions, f, indent=4)
        print(f"All records saved to {filename}")