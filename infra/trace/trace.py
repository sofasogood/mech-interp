#!/usr/bin/env python3
from transformer_lens import HookedTransformer
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="prompts.txt")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="acts.parquet")
    return p.parse_args()


def load_model():
    name = "gpt2"
    model = HookedTransformer.from_pretrained(name, device="cuda")
    tok = model.tokenizer  # TLens wraps HF tokenizer for you
    return model, tok


def make_hook(acc, tag):
    def _hook(tensor, hook=None):
        acc[tag].append(tensor.detach().cpu())

    return _hook


def forward(prompts):
    # 1️⃣ Load model + tokenizer on GPU
    model, tok = load_model()  # already lives on CUDA

    # 2️⃣ Create a dict to store activations **once per model instance**
    acts = {f"L{i}": [] for i in range(len(model.blocks))}

    # 3️⃣ Register one hook per layer.  This must happen **before** the
    #    first forward() so TLens knows to call your hook factory.
    for i in range(len(model.blocks)):
        model.blocks[i].hook_mlp_out.add_hook(make_hook(acts, f"L{i}"))

    # 4️⃣ Now it's safe to loop over prompts and run the model
    for batch in range(0, len(prompts), args.batch):
        sub = prompts[batch : batch + args.batch]
        toks = tok(sub, return_tensors="pt", padding=True)["input_ids"].to("cuda")
        with torch.no_grad():
            _ = model(toks)  # hooks fire here, filling `acts`

    return acts  # hand the collected tensors back


def to_parquet(acc, path):
    # Reshape each tensor to be 1D before converting to numpy
    cols = {k: torch.stack(v).half().reshape(-1).numpy() for k, v in acc.items()}
    table = pa.Table.from_pydict(cols)
    pq.write_table(table, path, compression="zstd")


if __name__ == "__main__":
    args = get_args()
    # prompts = pathlib.Path(args.prompts).read_text().splitlines()
    prompts = ["Goodfire's mission is", "Why is RLHF brittle?"]
    acts = forward(prompts)  # Capture the return value
    to_parquet(acts, args.out)
