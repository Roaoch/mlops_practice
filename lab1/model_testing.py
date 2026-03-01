import sys
import pickle
import torch

from datasets import Dataset
from sklearn.metrics import f1_score

sys.stdout = None

device = torch.device('cpu')

ds = Dataset.from_parquet('test/test.parquet')

trues = list(ds['is_duplicate'])

ds =  ds.select_columns(['embedding1', 'embedding2', 'is_duplicate'])
ds = ds.with_format('torch', device=device.type)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f).to(device)

preds = []
with torch.no_grad():
    for data in ds:
        emb_1, emb_2, target = data['embedding1'], data['embedding2'], data['is_duplicate']
        out = model(emb_1.reshape(1, -1), emb_2.reshape(1, -1))[0].detach().cpu().numpy()
        preds.append(
            1 if out[0] > 0.5 else 0
        )

f1 = f1_score(trues, preds)

sys.stdout = sys.__stdout__
print(f1)