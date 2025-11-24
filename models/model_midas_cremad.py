import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
# from PIL import video
from torchmetrics.classification import MulticlassCohenKappa
# from pytorch_lightning.metrics import functional as FM
from torchmetrics import functional as FM
from datasets.Cremad_dataset import CREMADDataset
import warnings
import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
# import fastaudio
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os
import copy
from .kinetics_resnet import resnet18

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

class AudioAndVisionModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        audio_module,
        video_module,
        audio_feature_dim,
        video_feature_dim,
        fusion_output_size,
        dropout_p,
        use_augmentation,
        train_dataset,
        batch_size = 32,
        precompute_features=False,
        device_idx=None,
        drop=None,
        warmup_epochs = 10
    ):
        super(AudioAndVisionModel, self).__init__()
        self.audio_module = audio_module
        self.video_module = video_module
        self.train_dataset = train_dataset
        # Calculate total input dimension for task_net
        print("audio_feature_dim", audio_feature_dim)
        print("video_feature_dim", video_feature_dim)
        total_feature_dim = audio_feature_dim + video_feature_dim
        self.task_net = torch.nn.Sequential(
            torch.nn.Linear(total_feature_dim, num_classes)
        )
        self.audio_classifier = torch.nn.Linear(
            audio_feature_dim, num_classes
        )
        self.video_classifier = torch.nn.Linear(
            video_feature_dim, num_classes
        )
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.loss_fn_none = torch.nn.CrossEntropyLoss(reduction='none')
        self.dropout = torch.nn.Dropout(dropout_p)
        self.use_augmentation = use_augmentation

        self.batch_size = batch_size
        self.alpha = 1.0
        self.warmup_epochs = warmup_epochs
    
    def forward(self, audio, video, label=None, use_augmentation=False, current_epoch = None):
        self.device = audio.device
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        audio_features = self.audio_module(audio)
        video_features = self.video_module(video)

        (_, C, H, W) = video_features.size()
        B = audio_features.size()[0]
        video_features = video_features.view(B, -1, C, H, W)
        video_features = video_features.permute(0, 2, 1, 3, 4)

        audio_features = F.adaptive_avg_pool2d(audio_features, 1)
        video_features = F.adaptive_avg_pool3d(video_features, 1)

        audio_features = torch.flatten(audio_features, 1)
        video_features = torch.flatten(video_features, 1)

        fusion_features = torch.cat([audio_features, video_features], dim=1)
        
        logits = self.task_net(fusion_features)
        loss = self.loss_fn(logits, label) if label is not None else label

        # Add augmentation losses if applicable
        if use_augmentation and label is not None:
            eta = 5e-2
            mis_loss_weight = 1
            audio_logits = self.audio_classifier(audio_features)
            video_logits = self.video_classifier(video_features)
            
            audio_probs = F.softmax(audio_logits.detach(), dim=1)
            video_probs = F.softmax(video_logits.detach(), dim=1)
            audio_conf = torch.gather(audio_probs, 1, label.unsqueeze(1)).squeeze()
            video_conf = torch.gather(video_probs, 1, label.unsqueeze(1)).squeeze()

            uni_loss_1 = self.loss_fn(audio_logits, label)
            uni_loss_2 = self.loss_fn(video_logits, label)
            uni_loss = (uni_loss_1 + uni_loss_2)/2
            if current_epoch < self.warmup_epochs:
                loss = uni_loss
                return logits, loss
            else:
                random_idx = torch.randperm(audio.size(0)).to(audio.device)
                mixed_loss = 0
                uni_ratio = audio_conf.mean() / video_conf.mean()

                lambd_1, lambd_2 = 1, 0
                mixed_features = torch.cat([lambd_1*audio_features + (1-lambd_1)*audio_features[random_idx], lambd_2*video_features + (1-lambd_2)*video_features[random_idx]], dim=1)
                mixed_logits = self.task_net(mixed_features)
                audio_weight = audio_conf / (audio_conf + video_conf[random_idx])
                video_weight = 1 - audio_weight
                mixed_audio_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label.unsqueeze(1)).squeeze()
                mixed_video_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label[random_idx].unsqueeze(1)).squeeze()
                mm_ratio_1 = mixed_audio_prob.detach().mean() / mixed_video_prob.detach().mean()
                if uni_ratio > 1:
                    weighted_mixed_loss = audio_weight * self.loss_fn_none(mixed_logits, label) + self.alpha * video_weight * self.loss_fn_none(mixed_logits, label[random_idx])
                else:
                    weighted_mixed_loss = self.alpha * audio_weight * self.loss_fn_none(mixed_logits, label) + video_weight * self.loss_fn_none(mixed_logits, label[random_idx])
                video_sim = torch.cosine_similarity(video_features, video_features[random_idx], dim=1).detach()
                mixed_loss += (1 + (video_sim + 1)/2) * weighted_mixed_loss

                lambd_1, lambd_2 = 0, 1
                mixed_features = torch.cat([lambd_1*audio_features + (1-lambd_1)*audio_features[random_idx], lambd_2*video_features + (1-lambd_2)*video_features[random_idx]], dim=1)
                mixed_logits = self.task_net(mixed_features)
                audio_weight = audio_conf[random_idx] / (audio_conf[random_idx] + video_conf)
                video_weight = 1 - audio_weight            
                mixed_audio_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label[random_idx].unsqueeze(1)).squeeze()
                mixed_video_prob = torch.gather(F.softmax(mixed_logits, dim=1), 1, label.unsqueeze(1)).squeeze()
                mm_ratio_2 = mixed_audio_prob.detach().mean() / mixed_video_prob.detach().mean()
                if uni_ratio > 1:
                    weighted_mixed_loss = audio_weight * self.loss_fn_none(mixed_logits, label[random_idx]) + self.alpha * video_weight * self.loss_fn_none(mixed_logits, label)
                else:   
                    weighted_mixed_loss = self.alpha * audio_weight * self.loss_fn_none(mixed_logits, label[random_idx]) + video_weight * self.loss_fn_none(mixed_logits, label)
                audio_sim = torch.cosine_similarity(audio_features, audio_features[random_idx], dim=1).detach()
                mixed_loss += (1 + (audio_sim + 1)/2) * weighted_mixed_loss
                
                mixed_loss = mixed_loss.mean()/2

                mm_ratio = (mm_ratio_1 + mm_ratio_2)/2
                if uni_ratio > 1:
                    if mm_ratio > uni_ratio:
                        self.alpha += eta
                    else:
                        self.alpha -= eta
                else:
                    if mm_ratio < uni_ratio:
                        self.alpha += eta
                    else:
                        self.alpha -= eta
                self.alpha = max(1, self.alpha)

                loss += mis_loss_weight * mixed_loss + uni_loss

                return logits, loss, self.alpha, audio_conf, video_conf, mixed_audio_prob, mixed_video_prob, video_sim, audio_sim
        return (logits, loss)

class MidasModel(pl.LightningModule):
    def __init__(self, params):
        for data_key in ["train_path", "dev_path", "audio_dir", "video_dir"]:
            # ok, there's one for-loop but it doesn't count
            if data_key not in params.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )
        
        super(MidasModel, self).__init__()
        self.model_params = params
        
        # assign some hparams that get used in multiple places
        self.embedding_dim = self.model_params.get("embedding_dim", 300)
        self.audio_feature_dim = self.model_params.get("audio_feature_dim", 300)
        # balance language and vision features by default
        self.video_feature_dim = self.model_params.get("video_feature_dim", 300)
        self.output_path = Path(self.model_params.get("output_path", "model-outputs"))
        self.output_path.mkdir(exist_ok=True)
        
        self.max_epochs = self.model_params.get("max_epochs", 100)
        # instantiate transforms, datasets
        self.audio_transform = self._build_audio_transform()
        self.video_transform = self._build_video_transform()
        self.train_dataset = self._build_dataset("train")
        # self.origin_train_dataset = self._build_dataset("train")
        self.origin_train_dataset = copy.deepcopy(self.train_dataset)
        self.dev_dataset = self._build_dataset("val")
        
        self.automatic_optimization = False
        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        self.use_augmentation = self.model_params.get("use_augmentation")

        self.warmup_epochs = self.model_params.get("warmup_epochs", 10)
        self.val_origin_acc = []
        self.val_swapped_acc = []
        self.val_swapped_conf_1 = []
        self.val_swapped_conf_2 = []
        self.alpha_per_epoch = []
        self.alpha = 0.0

        self.audio_conf_per_epoch = []
        self.video_conf_per_epoch = []
        self.audio_conf_per_epoch_multi = []
        self.video_conf_per_epoch_multi = []
        self.audio_conf_uni = 0.0
        self.video_conf_uni = 0.0
        self.audio_conf_multi = 0.0
        self.video_conf_multi = 0.0

        self.video_sim_per_epoch = []
        self.audio_sim_per_epoch = []

    @property
    def hparams(self):
        return self.model_params

    def forward(self, audio, video, label=None, use_augmentation=False, current_epoch=None):
        return self.model(audio, video, label, use_augmentation, current_epoch)

    def on_train_epoch_start(self):
        self.train_epoch_loss = 0.0
        self.current_epoch_video_sim = []
        self.current_epoch_audio_sim = []

    def training_step(self, batch, batch_nb):
        self.train()
        accumulate_grad_batches = self.hparams.get("accumulate_grad_batches", 1)
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        if self.use_augmentation:
            if self.current_epoch < self.warmup_epochs:
                _, loss = self.forward(
                audio=batch["audio"], 
                video=batch["video"], 
                label=batch["label"],
                use_augmentation=True,
                current_epoch=self.current_epoch,
            )
            else:
                _, loss, alpha, audio_conf_uni, video_conf_uni, audio_conf_multi, video_conf_multi, video_sim, audio_sim = self.forward(
                audio=batch["audio"], 
                video=batch["video"], 
                label=batch["label"],
                use_augmentation=True,
                current_epoch=self.current_epoch,
            )
                self.alpha = alpha
                self.audio_conf_uni += audio_conf_uni.mean().item()
                self.video_conf_uni += video_conf_uni.mean().item()
                self.audio_conf_multi += audio_conf_multi.mean().item()
                self.video_conf_multi += video_conf_multi.mean().item()

                if self.current_epoch >= self.warmup_epochs:
                    if not hasattr(self, 'current_epoch_video_sim'):
                        self.current_epoch_video_sim = []
                        self.current_epoch_audio_sim = []
                    self.current_epoch_video_sim.extend(video_sim.cpu().numpy().tolist())
                    self.current_epoch_audio_sim.extend(audio_sim.cpu().numpy().tolist())
        else:
            _, loss = self.forward(
                audio=batch["audio"], 
                video=batch["video"], 
                label=batch["label"]
                )
        loss = loss / accumulate_grad_batches

        self.manual_backward(loss)

        if (batch_nb + 1) % accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        if self.current_epoch >= self.warmup_epochs:
            if self.trainer.is_last_batch:
                sch.step()

        self.log("train_loss", loss, prog_bar=True)

        self.train_epoch_loss =  self.train_epoch_loss + loss.item()
        return {"train_loss": loss}
    
    def on_train_epoch_end(self):
        avg_train_loss = self.train_epoch_loss / len(self.train_dataloader())
        self.train_losses_per_epoch.append(avg_train_loss)
        self.train_epoch_loss = 0.0 
        self.log("avg_train_loss", avg_train_loss)

        self.audio_conf_uni = self.audio_conf_uni / len(self.train_dataloader())
        self.video_conf_uni = self.video_conf_uni / len(self.train_dataloader())
        self.audio_conf_multi = self.audio_conf_multi / len(self.train_dataloader())
        self.video_conf_multi = self.video_conf_multi / len(self.train_dataloader())
        self.audio_conf_per_epoch.append(self.audio_conf_uni)
        self.video_conf_per_epoch.append(self.video_conf_uni)
        self.audio_conf_per_epoch_multi.append(self.audio_conf_multi)
        self.video_conf_per_epoch_multi.append(self.video_conf_multi)
        print(f"audio_conf_uni: {self.audio_conf_uni:.2f}, audio_conf_multi: {self.audio_conf_multi:.2f}")
        print(f"video_conf_uni: {self.video_conf_uni:.2f}, video_conf_multi: {self.video_conf_multi:.2f}")
        self.audio_conf_uni = 0.0
        self.video_conf_uni = 0.0
        self.audio_conf_multi = 0.0
        self.video_conf_multi = 0.0
        
        self.alpha_per_epoch.append(self.alpha)
        print(f"alpha: {self.alpha:.2f}")

        if self.hparams.get("recording"):
            correct_original = 0
            correct_swapped = 0
            total_samples = 0
            conf_1_swapped = 0
            conf_2_swapped = 0
            with torch.no_grad():
                self.eval()
                for batch in self.trainer.val_dataloaders:
                    audio, video, labels = batch["audio"], batch["video"], batch["label"]
                    batch_size = audio.size(0)
                    audio = audio.to(self.device)
                    video = video.to(self.device)
                    labels = labels.to(self.device)
                    logits, _ = self.forward(audio, video, labels, use_augmentation=False)
                    _, pred = F.softmax(logits, dim=1).max(dim=1)
                    correct_original += (pred == labels).sum().item()

                    random_idx = torch.randperm(batch_size)
                    random_video = video[random_idx]
                    logits, _ = self.forward(audio, random_video, labels, use_augmentation=False)
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
            audio=batch["audio"], 
            video=batch["video"], 
            label=batch["label"]
        )
        
        # Get actual predictions by taking argmax
        preds = torch.argmax(logits, dim=1)
        
        acc = FM.accuracy(preds, batch["label"], task="multiclass", num_classes=self.hparams.get("num_classes", 6))
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        self.val_epoch_loss = self.val_epoch_loss + loss.item()
        return {"batch_val_loss": loss, "batch_val_acc": acc}

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_epoch_loss / len(self.val_dataloader())
        self.val_losses_per_epoch.append(avg_val_loss)

        self.val_epoch_loss = 0.0
        self.log("avg_val_loss", avg_val_loss)
        return {
            "val_loss": avg_val_loss
        }
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.get("lr", 0.001), momentum=0.9, weight_decay=self.hparams.get("weight_decay", 1e-4))
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
            'monitor': 'avg_val_loss',  
            'interval': 'epoch',    
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    # @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
            # collate_fn = collate_fn
        )

    # @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16),
            # collate_fn = collate_fn
        )
    
    ## Convenience Methods ##
    
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
        audio, video, labels = batch["audio"], batch["video"], batch["label"]
        preds, _ = self.forward(audio, video)
        preds = torch.argmax(preds, dim=1)

        acc = FM.accuracy(preds, labels, task="multiclass", num_classes=6)
        f1 = FM.f1_score(preds, labels, task="multiclass", num_classes=6, average="macro")

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        return {"test_acc": acc, "test_f1": f1}
    
    def on_test_epoch_end(self):
        # Access test_acc metrics directly through self.trainer.callback_metrics
        test_acc = self.trainer.callback_metrics["test_acc"]

        self.log("test_acc", test_acc)
        return {"test_acc": test_acc}

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

        test_acc = FM.accuracy(all_preds, all_labels, task="multiclass", num_classes=6)
        print(f"Test Accuracy: {test_acc.item():.4f}")
        return test_acc

    def _build_audio_transform(self):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        
    def _build_video_transform(self):
        video_dim = self.hparams.get("video_dim", 224)
        video_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(video_dim, video_dim)
                ),        
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return video_transform
    
    def _build_dataset(self, mode):
        return CREMADDataset(
            mode = mode,
            audio_path=self.hparams.get("audio_dir"),
            visual_path=self.hparams.get("video_dir"),
            train_csv = self.hparams.get("train_path"),
            val_csv = self.hparams.get("dev_path"),
            stat_csv = self.hparams.get("stat_path"),
        )
    
    def _build_model(self):
        audio_module = resnet18(modality = 'audio')
        video_module = resnet18(modality = 'visual')

        return AudioAndVisionModel(
            num_classes=self.hparams.get("num_classes", 6),
            loss_fn=torch.nn.CrossEntropyLoss(),
            audio_module=audio_module,
            video_module=video_module,
            audio_feature_dim=self.audio_feature_dim,
            video_feature_dim=self.video_feature_dim,
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
            warmup_epochs=self.hparams.get("warmup_epochs", 10)
        )
    
    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            filename='{epoch}-{val_loss:.3f}-{val_acc:.3f}',
            monitor='val_acc',  # Metric to monitor
            mode='max',         # Save model when metric is maximized
            save_top_k=1,      # Save only the best model
            verbose=True
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
            "callbacks": [checkpoint_callback, early_stop_callback],
            "devices": self.hparams.get("devices", [1]),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "num_sanity_val_steps": 0,
            "reload_dataloaders_every_n_epochs": 1
        }
        return trainer_params
    
    def save_records(self, seed):
        if len(self.val_origin_acc) == 0:
            return

        method = self.hparams.get("method", " ")
        filename = "save_records/cremad_record_" + method + "_" + str(seed) + ".json"
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
    
    def _save_similarity_histograms(self, epoch, video_sim_values, audio_sim_values):
        hist_dir = self.output_path / "histograms"
        hist_dir.mkdir(exist_ok=True)
        
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Video similarity histogram
        ax1.hist(video_sim_values, bins=20, range=(-1, 1), alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title(f'Video Similarity Distribution - Epoch {epoch}')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(-1, 1)
        ax1.grid(True, alpha=0.3)
        
        # Audio similarity histogram
        ax2.hist(audio_sim_values, bins=20, range=(-1, 1), alpha=0.7, color='red', edgecolor='black')
        ax2.set_title(f'Audio Similarity Distribution - Epoch {epoch}')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(-1, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        video_mean = np.mean(video_sim_values)
        video_std = np.std(video_sim_values)
        audio_mean = np.mean(audio_sim_values)
        audio_std = np.std(audio_sim_values)
        
        ax1.text(0.02, 0.98, f'Mean: {video_mean:.3f}\nStd: {video_std:.3f}\nSamples: {len(video_sim_values)}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.text(0.02, 0.98, f'Mean: {audio_mean:.3f}\nStd: {audio_std:.3f}\nSamples: {len(audio_sim_values)}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        filename = hist_dir / f"similarity_histogram_epoch_{epoch:03d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Similarity histogram saved: {filename}")

    def save_conf_records(self, seed):
        method = self.hparams.get("method", " ")
        filename = "save_records/all_cremad_conf_record_" + method + "_" + str(seed) + ".json"
        contributions = [
            {
                "epoch": i,
                "audio_conf_uni": self.audio_conf_per_epoch[i],
                "audio_conf_multi": self.audio_conf_per_epoch_multi[i],
                "video_conf_uni": self.video_conf_per_epoch[i],
                "video_conf_multi": self.video_conf_per_epoch_multi[i],
            }
            for i in range(len(self.alpha_per_epoch))
        ]

        with open(filename, "w") as f:
            json.dump(contributions, f, indent=4)
        print(f"All records saved to {filename}")