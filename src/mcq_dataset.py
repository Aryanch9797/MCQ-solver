""" This module defines the MCQDataset class, which is a PyTorch Dataset for handling multiple-choice question data.
It processes the input DataFrame to create pairs of questions and options, along with their corresponding labels for
training or evaluation."""


import torch
from torch.utils.data import Dataset
import pandas as pd

class MCQDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test
        self.options_list = ["A", "B", "C", "D", "E"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        question = row["prompt"]
        true_answer = None if self.is_test else row.get("answer", None)
        text_pairs, labels = [], []
        
        # Question and options are paired together, and labels are assigned based on whether the option is the true answer or not.
        for option in self.options_list:
            text_pairs.append([question, row[option]])
            labels.append(1.0 if option == true_answer else 0.0)
            
        return {"questions": text_pairs, "labels": torch.tensor(labels, dtype=torch.float32)}
