from typing import List, Any
import pickle
import pandas as pd


def save_file_pickle(fname: str, obj: Any) -> None:
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname: str) -> None:
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def chunkify(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    "turn pandas.dataframe into equal size n chunks."
    return [df[i::n] for i in range(n)]

# Copied from code2vec extractor.py to hash context_paths
def java_string_hashcode(s):
        """
        Imitating Java's String#hashCode
        """
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
