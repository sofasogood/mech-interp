import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, AutoTokenizer
import os


def main():
    # init the process group first
    torch.distributed.init_process_group("nccl")

    # torchrun injects this for you per-process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    # Set max length for tokenizer
    max_length = 512

    # Define preprocessing function for dataset
    def tokenize_function(examples):
        # Process the examples to return tensors instead of lists
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return outputs

    # Load and preprocess dataset
    ds = load_dataset("roneneldan/TinyStories", split="train[:2%]")
    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids"])

    # Create dataloader
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    model.config.pad_token_id = model.config.eos_token_id
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )

    model = model.to(device)
    model = FSDP(model, mixed_precision=mp, device_id=device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for step, batch in enumerate(dl):
        optim.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["input_ids"].to(device),
        )
        out.loss.backward()
        optim.step()
        if step % 20 == 0 and local_rank == 0:
            print(f"step {step}  loss {out.loss.item():.3f}")
        if step == 100:
            break

    # save sharded checkpoint
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        torch.save(model.state_dict(), f"ckpt_rank{local_rank}.pt")
    # Clean up process group
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
