import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Callable, Tuple


def make_collator(tokenizer, label_keys: Tuple[Any]=('labels_rationale', 'labels_output')) -> Callable:
    """Make collactor
    
    [Params]
    tokenizer
    label_keys : Tuple[Any] = ('labels_rationale', 'labels_output'))
    
    [Return]
    collator : Callable
    """
    pad_id = tokenizer.pad_token_id
    
    def _pad_label(seqs, seq_len: int) -> torch.Tensor:
        lab = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in seqs],
            batch_first=True,
            padding_value=-100,
        )
        
        cur_len = lab.size(1)
        if cur_len < seq_len:
            pad = lab.new_full((lab.size(0), seq_len - cur_len), -100)
            lab = torch.cat([lab, pad], dim=1)
        elif cur_len > seq_len:
            lab = lab[:, :seq_len]
        return lab
    
    def collate(batch):
        input_pad = tokenizer.pad(
            {'input_ids': [b['input_ids'] for b in batch]},
            padding=True, return_tensors='pt'
        )
        
        seq_len = input_pad['input_ids'].size(1)
        
        labels = {
            k : _pad_label([b[k] for b in batch], seq_len)
            for k in label_keys
        }
        
        return {
            'input_ids': input_pad['input_ids'],
            'attention_mask': input_pad['attention_mask'],
            **labels
        }
    
    return collate
