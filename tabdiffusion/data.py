# tabdiffusion/data.py
"""
Data handling: detect types, fit encoders/scalers, provide DataLoader-friendly dataset,
and inverse transform utils.
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset

class TabularPreprocessor:
    """
    Convert pandas DataFrame -> model-ready tensors and back.
    Automatically detects categorical columns (object / category) and numeric columns.
    It fits LabelEncoders for each categorical column and a StandardScaler for numerics.
    """

    def __init__(self, target_col: str = None, categorical_cols: Optional[List[str]] = None, numeric_cols: Optional[List[str]] = None):
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_encoders = {}   # name -> LabelEncoder
        self.scaler = None
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        # Auto-detect if needed
        if self.target_col is not None and self.target_col not in df.columns:
            raise ValueError("target_col not in df")

        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
            if self.target_col in self.categorical_cols:
                self.categorical_cols.remove(self.target_col)
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if self.target_col in self.numeric_cols:
                self.numeric_cols.remove(self.target_col)

        # Fit label encoders for categoricals (store classes)
        for c in self.categorical_cols:
            le = LabelEncoder()
            df[c] = df[c].astype(str).fillna("__NA__")
            le.fit(df[c].values)
            self.label_encoders[c] = le

        # Fit scaler for numeric columns
        if len(self.numeric_cols) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numeric_cols].fillna(0.0).values)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X_num, X_cat, y) as numpy arrays.
        X_num: (N, num_numeric)
        X_cat: (N, num_cat)
        y: (N,) - raw target values (not encoded) if target_col provided; for categorical target we provide encoded ints
        """
        assert self._fitted, "Call fit() first"
        df = df.copy()

        # Prepare numerics
        if len(self.numeric_cols) > 0:
            X_num = df[self.numeric_cols].fillna(0.0).values.astype(np.float32)
            X_num = self.scaler.transform(X_num)
        else:
            X_num = np.zeros((len(df), 0), dtype=np.float32)

        # Prepare cats as encoded ints
        if len(self.categorical_cols) > 0:
            X_cat = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
            for i, c in enumerate(self.categorical_cols):
                vals = df[c].astype(str).fillna("__NA__").values
                X_cat[:, i] = self.label_encoders[c].transform(vals)
        else:
            X_cat = np.zeros((len(df), 0), dtype=np.int64)

        # target handling
        y = None
        if self.target_col is not None:
            y_raw = df[self.target_col].values
            # encode target if categorical/text
            if df[self.target_col].dtype == "object" or df[self.target_col].dtype.name == "category":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                # store a temp encoder for target if needed (not persisted here)
            else:
                y = y_raw.astype(np.float32)
        return X_num, X_cat, y

    def inverse_transform(self, X_num: np.ndarray, X_cat: np.ndarray) -> pd.DataFrame:
        """
        Convert numeric array and categorical index array back to DataFrame with original column names & types.
        Numeric values will be inverse transformed to original scale if scaler fitted.
        Categorical indices will be inverse-transformed to original labels.
        """
        if len(self.numeric_cols) > 0 and self.scaler is not None:
            try:
                num_orig = self.scaler.inverse_transform(X_num)
            except Exception:
                num_orig = X_num
        else:
            num_orig = X_num

        out = {}
        for i, c in enumerate(self.numeric_cols):
            out[c] = num_orig[:, i] if num_orig.shape[1] > i else np.nan

        for i, c in enumerate(self.categorical_cols):
            if X_cat.shape[1] > i:
                idxs = X_cat[:, i].astype(int)
                le = self.label_encoders.get(c)
                if le is not None:
                    out[c] = le.inverse_transform(idxs)
                else:
                    out[c] = idxs
            else:
                out[c] = [None] * X_cat.shape[0]

        return pd.DataFrame(out)

class TabularDataset(Dataset):
    """
    Simple torch Dataset wrapper for preprocessed arrays (num, cat, y)
    """
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: Optional[np.ndarray] = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx], torch.tensor(0.0, dtype=torch.float32)
