"""
python train.py --model MODEL \
                --train input/train.csv \
                --valid input/valid.csv \
                --outdir ./output
"""
import os, argparse
from transformers import TrainingArguments
from datasets import disable_caching

from utils.config import load_config
from utils.data import load_split, make_collator
from utils.model import build_model, load_tokenizer
from multi_task_trainer import MultiTaskTrainer



config = load_config()
MODEL_REGISTRY = config['MODEL_REGISTRY']
ALPHA = config['ALPHA']


def run(model_key, train_csv, valid_csv, out_root, alpha):
    model_name = MODEL_REGISTRY[model_key]
    tokenizer = load_tokenizer(model_name)
    model = build_model(model_name, tokenizer)
    
    train_dataset = load_split(train_csv, tokenizer)
    valid_dataset = load_split(valid_csv, tokenizer)
    collator = make_collator(tokenizer)
    
    args = TrainingArguments(
        output_dir                  = out_root,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = 16,
        learning_rate               = 5e-5,
        max_steps                   = 1_500,
        warmup_steps                = 100,
        weight_decay                = 0.01,
        fp16                        = True,
        eval_strategy               = 'steps',
        eval_steps                  = 300,
        save_strategy               = 'steps',
        save_steps                  = 300,
        logging_steps               = 100,
        report_to                   = 'none',
        seed                        = 42,
        remove_unused_columns       = False,
        ###
        load_best_model_at_end      = True,
        metric_for_best_model       = 'eval_loss',
        greater_is_better           = False,
        label_names=["labels_rationale", "labels_output"]
    )
    
    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        alpha=alpha,
    )
    trainer.train()
    trainer.save_model(out_root)
    tokenizer.save_pretrained(out_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument('--train', default='input/train.csv')
    parser.add_argument('--valid', default='input/valid.csv')
    parser.add_argument('--outdir', default='./output')
    parser.add_argument('--alpha', type=float, default=ALPHA)
    args, _ = parser.parse_known_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    run(
        args.model, args.train, args.valid,
        os.path.join(args.outdir, args.model),
        alpha=args.alpha
    )
