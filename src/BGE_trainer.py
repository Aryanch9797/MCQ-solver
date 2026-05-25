import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedGroupKFold
from src.similar_questions_finder import find_similar_questions
from src.mcq_dataset import MCQDataset
from src.custom_collate import custom_collate_fn
import gc
import numpy as np


def bge_trainer_with_cv(model, test_loader, train_df, cv, epochs, lr, batch_size):

    sgkf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_best_epochs = []
    all_cv_logits = []  

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=train_df, y=train_df["answer"], groups=train_df["group_id"])):
        print(f"\n{'='*20} Fold {fold+1}/{cv} {'='*20}")
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset   = train_df.iloc[val_idx].reset_index(drop=True)
        
        # num_workers=2 helps CPU load data fast enough for dual GPUs
        train_loader = DataLoader(MCQDataset(train_subset), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
        val_loader   = DataLoader(MCQDataset(val_subset), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

        num_training_steps = len(train_loader) * epochs

        checkpoint_callback = ModelCheckpoint(
            dirpath="ensemble_checkpoints",
            filename=f"bge-fold{fold+1}-best",
            save_top_k=1,
            monitor="val_map",
            mode="max"
        )
        early_stop_callback = EarlyStopping(monitor="val_map", patience=3, mode="max")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=2,
            # strategy="ddp_notebook", # Required to prevent hanging in Kaggle Notebooks
            # precision="16-mixed",    # Enables Tensor Cores (FP16) for massive speed/VRAM improvements
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=CSVLogger("logs", name=f"fold_{fold+1}"),
            enable_progress_bar=True
        )
        trainer.fit(model, train_loader, val_loader)

        best_epoch = checkpoint_callback.best_model_score
        optimal_fold_epoch = trainer.early_stopping_callback.stopped_epoch - 3
        optimal_fold_epoch = max(1, optimal_fold_epoch) 
        fold_best_epochs.append(optimal_fold_epoch)

        print(f"[Fold {fold+1}] Running inference on test set...")
        predictions = trainer.predict(model, dataloaders=test_loader, ckpt_path="best")
        
        fold_logits = torch.cat(predictions, dim=0)
        all_cv_logits.append(fold_logits)

        # saving preds for ensemble
        torch.save(fold_logits.cpu(), f"bge_fold{fold+1}_logits.pt")

        del model, trainer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache() # Clear GPU VRAM between folds

    optimal_epochs = int(round(np.mean(fold_best_epochs)))
    print(f"\nCV Summary: Optimal epochs determined as {optimal_epochs}")