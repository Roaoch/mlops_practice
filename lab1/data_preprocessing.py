import datasets

import numpy as np

from datasets import Dataset
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

datasets.logging.set_verbosity(datasets.logging.CRITICAL)

def clear_ds(ds: Dataset) -> Dataset:
    ds = ds.filter(
        lambda row: row['question1'] is not None and \
            row['question2'] is not None and \
            len(row['question1']) > 5 and \
            len(row['question2']) > 5
    )

    ds = ds.map(
        lambda row: {
            'tokens1': simple_preprocess(row['question1']),
            'tokens2': simple_preprocess(row['question2'])
        }
    )

    return ds.filter(
        lambda row: len(row['tokens1']) > 2 and len(row['tokens2']) > 2
    )


def get_vectorizer(ds: Dataset):
    sents = [
        *ds['tokens1'],
        *ds['tokens2']
    ]
    
    return Word2Vec(
        sents,
        min_count=1,
        vector_size=128,
        window=5
    )


def get_vector(tokens: list[str], vectorizer: Word2Vec):
    res = []
    for w in tokens:
        if w in vectorizer.wv:
            res.append(vectorizer.wv[w])
        else:
            res.append([0. for _ in range(vectorizer.wv.vector_size)])
    res = np.array(res, dtype=np.float32)
    return res.mean(axis=0)


def preprocess(ds: Dataset):
    return ds.map(
        lambda row: {
            'embedding1': get_vector(row['tokens1'], vectorizer),
            'embedding2': get_vector(row['tokens2'], vectorizer),
        }
    )


train_ds = Dataset.from_parquet('train/train.parquet')
test_ds = Dataset.from_parquet('test/test.parquet')

train_ds = clear_ds(train_ds)
test_ds = clear_ds(test_ds)

vectorizer = get_vectorizer(train_ds)

train_ds = preprocess(train_ds)
test_ds = preprocess(test_ds)

train_ds.to_parquet('train/train.parquet')
test_ds.to_parquet('test/test.parquet')

train_ds.cleanup_cache_files()
test_ds.cleanup_cache_files()