""" This module defines a PyTorch Lightning model for training a BGE Reranker Large model on a multiple-choice question answering
task. The model is designed to compute the Mean Average Precision at 3 (MAP@3) metric during validation and uses a cosine learning
rate scheduler with warmup for optimization."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from src.mapat3 import compute_map_at_3


class BGERerankerLarge(pl.LightningModule):

    def __init__(self, lr, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        self.loss_fn = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.epochs = epochs
        self.lr = lr

    def forward(self, inputs):
        return self.model(**inputs, return_dict=True).logits.view(-1, 5)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets.argmax(dim=-1))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets.argmax(dim=-1))
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"logits": logits, "targets": targets})
        return loss
        
    def on_validation_epoch_end(self):
        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs], dim=0)
        
        val_map = compute_map_at_3(all_logits, all_targets)
        self.log("val_map", val_map, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear() 

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _ = batch
        return self(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        
        total_steps = self.epochs * self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
            }
        }

