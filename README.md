# tabdiffusion

TabDiffusion: conditional tabular diffusion generator (Transformer denoiser).

## Quickstart

```python
from tabdiffusion import TabDiffusion
import pandas as pd

df = pd.read_csv("your_table.csv")
td = TabDiffusion(df, target="isFraud", conditionals=["ProductCD","card4"])
td.fit(epochs=10, batch_size=256)

samples = td.sample(
    num_samples=500,
    labels_to_sample=[1,0],
    proportions=[0.7,0.3],
    feature_bias={"country":{"US":0.5,"CA":0.3,"NG":0.2}},
    steps=50
)
```

tabdiffusion stores training metrics in td.train_losses and td.val_losses, and saves the best checkpoint to tabdiffusion_best.pt by default.

