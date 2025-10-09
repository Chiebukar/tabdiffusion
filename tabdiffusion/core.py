# tabdiffusion/core.py
"""
High-level TabDiffusion class: user-facing API.
Provides:
- fit(df) or fit on init
- train(...)
- sample(...)
- stores metrics & checkpoints
"""

from .data import TabularPreprocessor, TabularDataset
from .model import TabDiffusionGenerator
from .sampler import generate_targeted_samples
from .utils import get_device, set_seed
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

class TabDiffusion:
    """
    Top-level API for training and sampling.
    """

    def __init__(
        self,
        df,
        target: str,
        conditionals: list = None,
        categorical_cols: list = None,
        numeric_cols: list = None,
        device="auto",
        random_seed: int = 42
    ):
        """
        Args:
            df: pandas DataFrame containing training data (will be copied)
            target: name of the target column
            conditionals: list of column names to use as condition signals (default: all categorical cols)
            categorical_cols / numeric_cols: optional overrides for auto-detection
            device: 'auto'|'cpu'|'cuda'
        """
        set_seed(random_seed)
        self.device = get_device(device)
        self.df = df.copy()
        self.target = target
        self.preprocessor = TabularPreprocessor(target_col=target, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
        self.preprocessor.fit(self.df)
        # default conditionals = all categorical columns (user can reduce)
        if conditionals is None:
            conditionals = self.preprocessor.categorical_cols.copy()
        self.conditionals = conditionals
        # build cond spec for model (include target as binary/categorical)
        cond_specs = {}
        # include target
        # if target is categorical in original df use card from label encoder else binary
        if self.df[self.target].dtype == "object" or self.df[self.target].dtype.name == "category":
            # treat target as categorical
            unique = self.df[self.target].astype(str).nunique()
            cond_specs[self.target] = {"type":"cat", "cardinality": unique}
        else:
            # assume binary/numeric
            cond_specs[self.target] = {"type":"binary"}
        # add other conditional categorical columns
        for c in self.conditionals:
            if c == self.target:
                continue
            # find cardinality from fitted label_encoders
            card = len(self.preprocessor.label_encoders[c].classes_)
            cond_specs[c] = {"type":"cat", "cardinality": card}

        self.cond_specs = cond_specs

        # model metadata
        self.num_num = len(self.preprocessor.numeric_cols)
        self.cat_cardinalities = [len(self.preprocessor.label_encoders[c].classes_) for c in self.preprocessor.categorical_cols]

        # placeholder attributes to be set on train
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.checkpoint_path = "tabdiffusion_best.pt"

    def build_model(self, **model_kwargs):
        """
        Instantiate the TabDiffusionGenerator with sensible defaults; user may override via model_kwargs.
        """
        defaults = dict(
            num_numeric=self.num_num,
            cat_cardinalities=self.cat_cardinalities,
            cond_columns=self.cond_specs,
            token_dim=192,
            time_embed_dim=128,
            transformer_layers=4,
            transformer_heads=4,
            transformer_ff=512,
            timesteps=100,
            uncond_prob=0.1
        )
        cfg = {**defaults, **model_kwargs}
        self.model = TabDiffusionGenerator(**cfg).to(self.device)
        return self.model

    def fit(
        self,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 2e-4,
        val_split: float = 0.2,
        checkpoint_path: str = "tabdiffusion_best.pt",
        model_kwargs: dict = None,
        trainer_kwargs: dict = None
    ):
        """
        Train the generator on self.df.
        Stores train_losses & val_losses on the object and saves best checkpoint to checkpoint_path.
        """
        model_kwargs = model_kwargs or {}
        self.build_model(**model_kwargs)
        checkpoint_path = checkpoint_path or self.checkpoint_path
        self.checkpoint_path = checkpoint_path

        # prepare data
        X_num, X_cat, y = self.preprocessor.transform(self.df)
        # If target is categorical, ensure y is integer labels
        if y is not None and y.dtype != np.float32 and y.dtype != np.float64:
            y = y.astype(np.int64)
        dataset = TabularDataset(X_num, X_cat, y)
        n = len(dataset)
        n_val = int(n * val_split)
        n_train = n - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

        best_val = float("inf")
        patience = trainer_kwargs.get("patience", 10) if trainer_kwargs else 10
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []
        epochs = int(epochs)
        for ep in range(1, epochs+1):
            # train epoch
            self.model.train()
            running = []
            for x_num, x_cat, ybatch in train_loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                # build cond batch: include target + selected conditionals
                cond_batch = {}
                # target: may be ybatch floats/ints
                cond_batch[self.target] = ybatch.to(self.device).long()
                for c in self.conditionals:
                    if c == self.target:
                        continue
                    idx = self.preprocessor.categorical_cols.index(c)
                    cond_batch[c] = x_cat[:, idx]
                loss = self.model.training_loss(x_num, x_cat, cond_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running.append(loss.item())
            train_loss = float(np.mean(running)) if len(running)>0 else 0.0
            self.train_losses.append(train_loss)

            # validation
            self.model.eval()
            vals = []
            with torch.no_grad():
                for x_num, x_cat, ybatch in val_loader:
                    x_num = x_num.to(self.device)
                    x_cat = x_cat.to(self.device)
                    cond_batch = {}
                    cond_batch[self.target] = ybatch.to(self.device).long()
                    for c in self.conditionals:
                        if c == self.target:
                            continue
                        idx = self.preprocessor.categorical_cols.index(c)
                        cond_batch[c] = x_cat[:, idx]
                    vloss = self.model.training_loss(x_num, x_cat, cond_batch)
                    vals.append(vloss.item())
            val_loss = float(np.mean(vals)) if len(vals)>0 else 0.0
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # checkpoint
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"[Epoch {ep}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if patience_counter >= (trainer_kwargs.get("early_stopping", 5) if trainer_kwargs else 5):
                print("Early stopping triggered.")
                break

        # load best
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def sample(
        self,
        num_samples: int = 100,
        labels_to_sample: list = None,
        proportions: list = None,
        feature_bias: dict = None,
        cond_overrides: dict = None,
        steps: int = 50,
        cfg_scale: float = 1.5,
        device="auto",
        scaler=None
    ):
        """
        High level sample function.
        - labels_to_sample: list of labels (ints) corresponding to the target column. If None, defaults to [unique target values].
        - proportions: list of floats summing to 1 for the labels_to_sample.
        - feature_bias: dict of form {"country": {"US":0.5, "CA":0.3, "NG":0.2}} to bias sampling for specific categorical features.
        - cond_overrides: additional unconditional overrides applied to all generated samples.
        Returns: pd.DataFrame in original schema including target column.
        """
        device = get_device(device)
        if self.model is None:
            raise RuntimeError("Model not built/trained. Call fit() first or call build_model() then load a checkpoint.")

        # labels
        if labels_to_sample is None:
            # try infer unique labels from df
            labels_unique = np.unique(self.df[self.target].values)
            labels_to_sample = labels_unique.tolist()
        labels_to_sample = list(labels_to_sample)

        if proportions is None:
            # if two labels default to proportion 0.5 split
            if len(labels_to_sample) == 1:
                proportions = [1.0]
            elif len(labels_to_sample) == 2:
                proportions = [0.5, 0.5]
            else:
                # uniform
                proportions = [1.0 / len(labels_to_sample)] * len(labels_to_sample)

        # cond_overrides + feature_bias - incorporate feature_bias into cond_overrides by sampling categorical values per requested proportion
        cond_overrides = cond_overrides or {}
        # If user provided feature_bias, we will sample values according to specified proportions for those columns and fill cond_overrides with vectors
        co = dict(cond_overrides)
        if feature_bias:
            for col, pm in feature_bias.items():
                # pm is dict mapping label-> proportion
                labels = list(pm.keys())
                probs = np.array(list(pm.values()), dtype=float)
                probs = probs / probs.sum()
                chosen = np.random.choice(labels, size=num_samples, p=probs)
                # map chosen labels to encoded ints
                if col in self.preprocessor.label_encoders:
                    le = self.preprocessor.label_encoders[col]
                    encoded = le.transform([str(v) for v in chosen])
                    co[col] = encoded
                else:
                    # pass raw chosen values
                    co[col] = chosen

        # call generate_targeted_samples (sampler ensures cond_batch shape and calls model.sample)
        df_out = generate_targeted_samples(
            gen=self.model,
            num_samples=num_samples,
            target_col=self.target,
            labels_to_sample=labels_to_sample,
            proportions=proportions,
            cond_overrides=co,
            steps=steps,
            cfg_scale=cfg_scale,
            device=self.device,
            preprocessor=self.preprocessor
        )
        return df_out
