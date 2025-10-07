# tabdiffusion/sampler.py
"""
Sampling helper that provides high-level functions to build cond_batch,
perform target-distribution sampling, feature-biasing, and return DataFrame.
"""

import torch
import numpy as np
import pandas as pd

def _build_cond_batch(gen, num_samples, cond_overrides, device):
    """
    Build cond_batch dict (col -> tensor of shape (num_samples,))
    cond_overrides may contain scalar value or list/ndarray (length==num_samples).
    """
    cond_batch = {}
    for col, spec in gen.cond_specs.items():
        typ = spec["type"]
        if cond_overrides and col in cond_overrides:
            val = cond_overrides[col]
            if isinstance(val, (list, np.ndarray, torch.Tensor)):
                arr = np.array(val)
                if arr.shape[0] != num_samples:
                    raise ValueError(f"cond override length for {col} must equal num_samples or be scalar")
                if typ in ("cat","binary"):
                    cond_batch[col] = torch.tensor(arr, dtype=torch.long, device=device)
                else:
                    cond_batch[col] = torch.tensor(arr, dtype=torch.float32, device=device)
            else:
                if typ in ("cat","binary"):
                    cond_batch[col] = torch.full((num_samples,), int(val), dtype=torch.long, device=device)
                else:
                    cond_batch[col] = torch.full((num_samples,), float(val), dtype=torch.float32, device=device)
        else:
            # default zeros (classifier-free guidance may mask cond)
            if typ in ("cat","binary"):
                cond_batch[col] = torch.zeros(num_samples, dtype=torch.long, device=device)
            else:
                cond_batch[col] = torch.zeros(num_samples, dtype=torch.float32, device=device)
    return cond_batch

def generate_targeted_samples(gen, num_samples, target_col, labels_to_sample, proportions, cond_overrides, steps, cfg_scale, device, preprocessor):
    """
    Generate samples split across labels_to_sample per proportions (list sums to 1).
    Returns DataFrame in original schema including target column.
    """
    device = torch.device(device)
    parts = []
    # Determine counts
    counts = []
    if len(labels_to_sample) == 1:
        counts = [num_samples]
    else:
        counts = [int(num_samples * p) for p in proportions]
        # adjust rounding
        diff = num_samples - sum(counts)
        for i in range(diff):
            counts[i % len(counts)] += 1

    for label, cnt in zip(labels_to_sample, counts):
        if cnt <= 0:
            continue
        # build overrides plus target
        co = dict(cond_overrides or {})
        co[target_col] = int(label)
        cond_batch = _build_cond_batch(gen, cnt, co, device)
        x_num_gen, x_cat_gen = gen.sample(cnt, cond_batch, steps=steps, cfg_scale=cfg_scale)
        # inverse transform to DataFrame rows
        X_num_np = x_num_gen.detach().cpu().numpy()
        X_cat_np = x_cat_gen.detach().cpu().numpy() if x_cat_gen.numel() > 0 else np.zeros((cnt, 0), dtype=int)
        df_part = preprocessor.inverse_transform(X_num_np, X_cat_np)
        # attach target (preserve original raw label if preprocessor target permutes—here we keep numeric)
        df_part[target_col] = label
        parts.append(df_part)
    if len(parts) == 0:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
