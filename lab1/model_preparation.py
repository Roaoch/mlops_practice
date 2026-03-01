import sys
import torch
import pickle
import datasets

from torch.utils.data import DataLoader
from datasets import Dataset

from __modeling import SiameseNet

sys.stdout = None
datasets.logging.set_verbosity(datasets.logging.CRITICAL)

device = torch.device('cpu')

ds = Dataset.from_parquet('train/train.parquet')

ds =  ds.select_columns(['embedding1', 'embedding2', 'is_duplicate'])
ds = ds.with_format('torch', device=device.type)

dataloader = DataLoader(
    ds,
    shuffle=True
)

model = SiameseNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)

for _ in range(3):
    for data in dataloader:
        emb_1, emb_2, target = data['embedding1'], data['embedding2'], data['is_duplicate']
        target = target.reshape(1, -1).float()

        optimizer.zero_grad()

        out = model(emb_1, emb_2)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)