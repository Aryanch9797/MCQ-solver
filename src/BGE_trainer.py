import torch
import pandas as pd
import numpy as np
import gc
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedGroupKFold
from src.mcq_dataset import MCQDataset
from src.custom_collate import custom_collate_fn

def model_trainer_cv(model_class, model_name_str, test_loader, train_df, test_df, cv, epochs, lr, batch_size):
    """
    model_class: The uninstantiated PyTorch Lightning model class (e.g., BGERerankerLarge)
    model_name_str: String identifier for saving CSVs (e.g., 'deberta_v3_large')
    """
    sgkf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_best_epochs = []
    
    # Initialize tensors to hold OOF predictions and accumulated test predictions
    oof_logits = torch.zeros((len(train_df), 5), dtype=torch.float32)
    test_logits_sum = None

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=train_df, y=train_df["answer"], groups=train_df["group_id"])):
        print(f"\n{'='*20} Fold {fold+1}/{cv} {'='*20}")
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset   = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_loader = DataLoader(MCQDataset(train_subset), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
        val_loader   = DataLoader(MCQDataset(val_subset), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

        # Instantiate a fresh model for each fold to prevent weights leaking between folds
        model = model_class(lr, epochs)

        checkpoint_callback = ModelCheckpoint(
            dirpath="ensemble_checkpoints",
            filename=f"{model_name_str}-fold{fold+1}-best",
            save_top_k=1,
            monitor="val_map",
            mode="max"
        )
        early_stop_callback = EarlyStopping(monitor="val_map", patience=3, mode="max")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=2,
            strategy="ddp_notebook", 
            precision="16-mixed",    
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=CSVLogger("logs", name=f"{model_name_str}_fold_{fold+1}"),
            enable_progress_bar=True
        )
        
        # Train the model
        trainer.fit(model, train_loader, val_loader)

        optimal_fold_epoch = trainer.early_stopping_callback.stopped_epoch - 3
        fold_best_epochs.append(max(1, optimal_fold_epoch))

        # ---------------------------------------------------------
        # GENERATE OOF PREDICTIONS (Predicting on Val Set)
        # ---------------------------------------------------------
        print(f"[Fold {fold+1}] Generating OOF predictions...")
        val_preds = trainer.predict(model, dataloaders=val_loader, ckpt_path="best")
        val_fold_logits = torch.cat(val_preds, dim=0).cpu()
        
        # Map the predictions back to their original indices in train_df
        oof_logits[val_idx] = val_fold_logits

        # ---------------------------------------------------------
        # GENERATE TEST PREDICTIONS
        # ---------------------------------------------------------
        print(f"[Fold {fold+1}] Running inference on test set...")
        test_preds = trainer.predict(model, dataloaders=test_loader, ckpt_path="best")
        test_fold_logits = torch.cat(test_preds, dim=0).cpu()

        # Accumulate test predictions
        if test_logits_sum is None:
            test_logits_sum = test_fold_logits
        else:
            test_logits_sum += test_fold_logits

        # Cleanup to prevent VRAM OOM errors
        del model, trainer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Average the test predictions across all folds
    test_logits_avg = test_logits_sum / cv

    # ---------------------------------------------------------
    # SAVE OOF AND TEST LOGITS TO CSV
    # ---------------------------------------------------------
    print("\n[Saving OOF and Test Predictions]")
    
    # Save OOF
    oof_df = pd.DataFrame(oof_logits.numpy(), columns=["logit_A", "logit_B", "logit_C", "logit_D", "logit_E"])
    if "ID" in train_df.columns:
        oof_df.insert(0, "ID", train_df["ID"])
    # Append true answers to make meta-modeling easier later
    oof_df["answer"] = train_df["answer"]
    oof_df.to_csv(f"oof_pred_csv_{model_name_str}.csv", index=False)

    # Save Test Average
    test_df_out = pd.DataFrame(test_logits_avg.numpy(), columns=["logit_A", "logit_B", "logit_C", "logit_D", "logit_E"])
    if "ID" in test_df.columns:
        test_df_out.insert(0, "ID", test_df["ID"])
    test_df_out.to_csv(f"test_pred_csv_{model_name_str}.csv", index=False)

    optimal_epochs = int(round(np.mean(fold_best_epochs)))
    print(f"\nCV Summary: Optimal epochs determined as {optimal_epochs}")
    
    return optimal_epochs