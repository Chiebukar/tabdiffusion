# tabdiffusion/utils.py
"""
Utilities: device, seed, cosine schedule, plotting helpers.
"""

import math
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def get_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Cosine schedule of betas (from Nichol & Dhariwal improved DDPM).
    Returns betas as numpy array of size timesteps.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = []
    for t in range(1, steps):
        a_t = alphas_cumprod[t]
        a_t_1 = alphas_cumprod[t - 1]
        beta = min(1 - a_t / a_t_1, 0.999)
        betas.append(beta)
    return np.array(betas, dtype=np.float32)

def plot_losses(train_losses, val_losses, title="Loss"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
