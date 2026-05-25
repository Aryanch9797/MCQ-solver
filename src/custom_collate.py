
""" Custom collate function for DataLoader. This module defines a custom collate function to be used with PyTorch's DataLoader,
which processes batches of question pairs and their corresponding labels for training a model. The function handles tokenization
and padding of input sequences, ensuring that they are properly formatted for model input while optimizing for GPU performance.
"""

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')

def custom_collate_fn(batch):
    flat_pairs, labels = [], []
    for item in batch:
        flat_pairs.extend(item["questions"])
        labels.append(item["labels"])

    # GPU OPTIMIZATION: Reverted to padding=True for dynamic sequence lengths
    inputs = tokenizer(
        text=[p[0] for p in flat_pairs],
        text_pair=[p[1] for p in flat_pairs],
        padding=True, 
        truncation="only_first",
        return_tensors="pt", 
        max_length=512
    )
    targets = torch.stack(labels)
    return inputs, targets