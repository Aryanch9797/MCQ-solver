""" Evaluation metrics for MCQ solver. This module defines functions to compute evaluation metrics such as Mean Average Precision
at 3 (MAP@3) for the predictions made by the MCQ solver. """

def compute_map_at_3(predictions, labels):
    true_indices   = labels.argmax(dim=-1)
    ranked_indices = predictions.argsort(dim=-1, descending=True)
    map_score = 0.0
    for true_idx, ranked_row in zip(true_indices, ranked_indices):
        for rank, pred_idx in enumerate(ranked_row[:3]):
            if pred_idx == true_idx:
                map_score += 1.0 / (rank + 1)
                break
    return map_score / len(labels)