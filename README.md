# clip-lightning

Use clip with pytorch-lightning

## Usage

1. Firstly, clone the repo
```
git clone https://github.com/lianyongxing/clip-lightning.git
cd clip-lightning
```

2. Model Train
```
python train.py
```

3. Test

```
import clip
from wrapper import CLIPWrapper
import torch

model = CLIPWrapper()
checkpoint = "epoch=1-step=4.ckpt"
model.load_state_dict(torch.load(checkpoint)['state_dict'])

model.forward(text=clip.tokenize(['description for paper']))[:10
```