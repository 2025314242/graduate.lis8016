import torch, pandas as pd, re, argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.config import load_config

TAG_OUTPUT = re.compile(r"<output>(.*?)</output>", re.DOTALL)
MODEL_REGISTRY = load_config()["MODEL_REGISTRY"]

# ── model / tokenizer 로더 ────────────────────────────────────────
def load_best_model(ckpt_dir, base):
    tok  = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    # tok.padding_side = "left"
    close_id = tok.convert_tokens_to_ids("</output>")
    tok.eos_token_id = close_id
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_REGISTRY[base], trust_remote_code=True)
    base_model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base_model, ckpt_dir).eval()
    model.config.eos_token_id = close_id
    model.half()                       # fp16  ↗속도 / ↘VRAM
    return tok, model

# ── 배치 추론 ─────────────────────────────────────────────────────
@torch.inference_mode()
def batch_generate(tokenizer, model, prompts, max_new=256, bs=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in tqdm(range(0, len(prompts), bs), desc="Infer"):
        batch = prompts[i:i+bs]
        batch = [f"{p}<rationale></rationale><output>" for p in batch]

        enc = tokenizer(batch, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=4096).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        dec = tokenizer.batch_decode(out, skip_special_tokens=False)
        for txt in dec:
            txt = txt.replace("[|endofturn|]", "</output>")
            m = TAG_OUTPUT.search(txt)
            results.append(
                (m.group(1) if m else txt.split("<output>",1)[1]).strip()
            )
    return results

# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base",  required=True)
    ap.add_argument("--data",  required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--bs",      type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    prompts = df["input"].astype(str).tolist()

    tok, mdl = load_best_model(args.ckpt, args.base)
    outs = batch_generate(tok, mdl, prompts, args.max_new, args.bs)

    pd.DataFrame({"input": prompts, "output": outs}).to_csv(args.out, index=False)
    print(f"✔ saved: {args.out} ({len(outs)})")

if __name__ == "__main__":
    main()
