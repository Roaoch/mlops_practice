import os
import datasets

import numpy as np

from datasets import load_dataset

datasets.logging.set_verbosity(datasets.logging.CRITICAL)

ds = load_dataset("AlekseyKorshuk/quora-question-pairs", split='train')

# Немного обрезал датасет, чтобы оно обучалось не так долго
ds = ds.select(np.random.randint(0, len(ds), 50_000))

ds = ds.train_test_split()

if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('test'):
    os.mkdir('test')

ds['train'].to_parquet('train/train.parquet')
ds['test'].to_parquet('test/test.parquet')

ds.cleanup_cache_files()
