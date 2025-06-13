import pandas as pd
from datasets import Dataset

from ._encode_instance import encode_instance


def load_split(path: str, tokenizer, max_len: int=4096):
    """Load split
    
    [Params]
    path    : str
    tokenizer
    max_len : int
    
    [Return]
    split_set
    """
    df = pd.read_csv(path)
    split_set = Dataset.from_pandas(df).map(
        lambda r: encode_instance(r, tokenizer=tokenizer, max_len=max_len), # {input_ids, labels_rationale, labels_output}
        remove_columns=df.columns.tolist(),
        load_from_cache_file=False
    )
    
    return split_set
