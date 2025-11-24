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
from datasets.Ucf101_dataset import UCF101Dataset
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
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)
# torch.autograd.set_detect_anomaly(True)

class FlowAndVisionModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        flow_module,
        vision_module,
        flow_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        train_dataset,
        batch_size = 32,
        device_idx=None,
        drop=None
    ):
        super(FlowAndVisionModel, self).__init__()
        self.flow_module = flow_module
        self.vision_module = vision_module
        self.train_dataset = train_dataset
        
        # Calculate total input dimension for task_net
        print("flow_feature_dim", flow_feature_dim)
        print("vision_feature_dim", vision_feature_dim)
        total_feature_dim = flow_feature_dim + vision_feature_dim
        # total_feature_dim = 1024
        self.task_net = torch.nn.Sequential(
            torch.nn.Linear(total_feature_dim, num_classes),
        )
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.loss_fn_none = torch.nn.CrossEntropyLoss(reduction='none')
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, flow, image, label=None): #flow dim (B, 3, 2, 224, 224), image dim (B, 3, 3, 224, 224)
        B_f = len(flow)
        B_v = len(image)
        flow = flow.permute(0, 2, 1, 3, 4).contiguous()
        # flow = flow.contiguous().float()
        image = image.permute(0, 2, 1, 3, 4).contiguous()

        flow_features = self.flow_module(flow)
        image_features = self.vision_module(image)

        (_, C_f, H_f, W_f) = flow_features.size()
        (_, C_v, H_v, W_v) = image_features.size()

        flow_features = flow_features.view(B_f, -1, C_f, H_f, W_f).permute(0, 2, 1, 3, 4)
        image_features = image_features.view(B_v, -1, C_v, H_v, W_v).permute(0, 2, 1, 3, 4)

        flow_features = F.adaptive_avg_pool3d(flow_features, 1)
        image_features = F.adaptive_avg_pool3d(image_features, 1)

        flow_features = torch.flatten(flow_features, 1)
        image_features = torch.flatten(image_features, 1)

        fusion_features = torch.cat([flow_features, image_features], dim=1)
        
        logits = self.task_net(fusion_features)
        
        loss = (
            self.loss_fn(logits, label) 
            if label is not None else label
        )      
  
        return (logits, loss)


class BaselineModel(pl.LightningModule):
    def __init__(self, params):
        for data_key in ["train_path", "dev_path", "visual_path", "flow_path_u", "flow_path_v" ]:
            if data_key not in params.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )
        
        super(BaselineModel, self).__init__()
        self.model_params = params
        
        self.embedding_dim = self.model_params.get("embedding_dim", 300)
        self.flow_feature_dim = self.model_params.get("flow_feature_dim", 300)
        self.vision_feature_dim = self.model_params.get("vision_feature_dim", 300)
        self.output_path = Path(self.model_params.get("output_path", "model-outputs"))
        self.output_path.mkdir(exist_ok=True)
        
        self.max_epochs = self.model_params.get("max_epochs", 100)
        # self.flow_transform = self._build_flow_transform()
        # self.vision_transform = self._build_vision_transform()
        self.train_dataset = self._build_dataset("train")
        # self.origin_train_dataset = self._build_dataset("train")
        self.origin_train_dataset = copy.deepcopy(self.train_dataset)
        self.dev_dataset = self._build_dataset("val")
        
        self.automatic_optimization = False
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        
        self.val_origin_acc = []
        self.val_swapped_acc = []
        self.val_swapped_conf_1 = []
        self.val_swapped_conf_2 = []
        self.alpha_per_epoch = []
        self.alpha = 0.0
        
    @property
    def hparams(self):
        return self.model_params
    
    def forward(self, flow, image, label=None):
        return self.model(flow, image, label)
    
    def on_train_epoch_start(self):
        self.train_epoch_loss = 0.0

    def training_step(self, batch, batch_nb):
        self.train()
        accumulate_grad_batches = self.hparams.get("accumulate_grad_batches", 1)
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        # print("INPUT FLOW SHAPE:", batch["flow"].shape)  
        # print("INPUT IMAGE SHAPE:", batch["visual"].shape) 
        _, loss = self.forward(
            flow=batch["flow"], 
            image=batch["visual"], 
            label=batch["label"],
            )
        loss = loss / accumulate_grad_batches

        self.manual_backward(loss)

        if (batch_nb + 1) % accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        if self.trainer.is_last_batch:
            sch.step()

        self.log("train_loss", loss, prog_bar=True)

        self.train_epoch_loss =  self.train_epoch_loss + loss.item()
        return {"train_loss": loss}
    
    def on_train_epoch_end(self):
        # for batch in self.train_dataloader():
        #     print(f"Batch keys in train_dataloader: {batch.keys()}")
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
                    flow, image, labels = batch["flow"], batch["visual"], batch["label"]
                    batch_size = flow.size(0)
                    flow = flow.to(self.device)
                    image = image.to(self.device)
                    labels = labels.to(self.device)
                    logits, _ = self.forward(flow, image, labels)
                    _, pred = F.softmax(logits, dim=1).max(dim=1)
                    correct_original += (pred == labels).sum().item()

                    random_idx = torch.randperm(batch_size)
                    random_image = image[random_idx]
                    logits, _ = self.forward(flow, random_image, labels)
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
            flow=batch["flow"], 
            image=batch["visual"], 
            label=batch["label"],
        )
        
        # Get actual predictions by taking argmax
        preds = torch.argmax(logits, dim=1)
        
        acc = FM.accuracy(preds, batch["label"], task="multiclass", num_classes=self.hparams.get("num_classes",101))
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
        # Separate parameters for main network and augnet
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.001))
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.get("lr", 0.001), momentum=0.9, weight_decay=self.hparams.get("weight_decay", 1e-4))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        dataset = self.origin_train_dataset
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16),
            pin_memory=True,
            persistent_workers=True,
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
        flow, image, labels = batch["flow"], batch["visual"], batch["label"]
        preds, _ = self.forward(flow, image)
        preds = torch.argmax(preds, dim=1)

      
        acc = FM.accuracy(preds, labels, task="multiclass", num_classes=101)
        f1 = FM.f1_score(preds, labels, task="multiclass", num_classes=101, average="macro")

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

         및 F1 score 계산
        test_acc = FM.accuracy(all_preds, all_labels, task="multiclass", num_classes=101)
        test_f1 = FM.f1_score(all_preds, all_labels, task="multiclass", num_classes=101, average="macro")

        print(f"Test Accuracy: {test_acc.item():.4f}")
        print(f"Test F1 Score: {test_f1.item():.4f}")

        return test_acc, test_f1
    
    def _build_dataset(self, mode):

        return UCF101Dataset(
            mode = mode,
            stat_path = self.hparams.get("stat_path"),
            visual_path=self.hparams.get("visual_path"),
            flow_path_u = self.hparams.get("flow_path_u"),
            flow_path_v = self.hparams.get("flow_path_v"),
            train_path = self.hparams.get("train_path"),
            val_path = self.hparams.get("dev_path"),
        )
    
    def _build_model(self):

        flow_module = resnet18(modality = 'flow')
        vision_module = resnet18(modality = 'visual')

        return FlowAndVisionModel(
            num_classes=self.hparams.get("num_classes", 101),
            loss_fn=torch.nn.CrossEntropyLoss(),
            flow_module=flow_module,
            vision_module=vision_module,
            flow_feature_dim=self.flow_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size", 512
            ),
            dropout_p=self.hparams.get("dropout_p", 0.5),
            train_dataset=self.train_dataset,
            batch_size=self.hparams.get("batch_size", 32),
            device_idx=self.hparams.get("devices", [1]),
            drop=None
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
            # "accumulate_grad_batches": self.hparams.get(
            #     "accumulate_grad_batches", 1
            # ),
            # "devices": self.hparams.get("n_gpu", 1),
            "devices": self.hparams.get("devices", [1]),
            "max_epochs": self.hparams.get("max_epochs", 100),
            # "gradient_clip_val": self.hparams.get(
            #     "gradient_clip_value", 3
            # ),
            "num_sanity_val_steps": 0,
            "reload_dataloaders_every_n_epochs": 1
        }
        return trainer_params
    
    def save_records(self, seed):
        # Save the model
        if len(self.val_origin_acc) == 0:
            return

        method = self.hparams.get("method", " ")
        filename = "save_records/ucf101_record_" + method + "_" + str(seed) + ".json"
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