import torch
from transformers import Trainer


class MultiTaskTrainer(Trainer):
    """MultiTaskTrainer
    
    loss = lambda * L_output + (1 - lambda) * L_rationale
    """
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.label_names = ['labels_rationale', 'labels_output']
        self.args.label_names = self.label_names
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lbl_r = inputs.pop('labels_rationale')
        lbl_o = inputs.pop('labels_output')
        
        out = model(**inputs)
        logits = out.logits  # (B, T, V)
        
        shift_r = lbl_r[..., 1:].contiguous()     # (B, T-1)
        shift_o = lbl_o[..., 1:].contiguous()
        logits  = logits[..., :-1, :].contiguous()  # (B, T-1, V)
        
        # --- Masked loss: rationale ---
        mask_r = shift_r.ne(-100)
        if mask_r.any():
            logits_r = logits[mask_r]
            shift_r = shift_r[mask_r]
            loss_r = self.ce(logits_r, shift_r)
        else:
            loss_r = torch.tensor(0.0, device=logits.device)

        # --- Masked loss: output ---
        mask_o = shift_o.ne(-100)
        if mask_o.any():
            logits_o = logits[mask_o]
            shift_o = shift_o[mask_o]
            loss_o = self.ce(logits_o, shift_o)
        else:
            loss_o = torch.tensor(0.0, device=logits.device)

        loss = self.alpha * loss_o + (1 - self.alpha) * loss_r
        
        if return_outputs:
            return loss, out
        else:
            return loss
