from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import torch

model_name = "gpt2"  # fits in RAM instantly
tok = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name, device="cuda")

prompt = "Goodfire’s mission is"
inputs = tok(prompt, return_tensors="pt")["input_ids"].to("cuda")

# Collect top-5 neuron activations from the last MLP layer
acts = {}


def save_acts(module, inp, out):
    acts["mlp_out"] = out.detach().flatten()


model.blocks[-1].mlp.register_forward_hook(save_acts)
with torch.inference_mode():
    logits = model(inputs)
    next_ids = logits[:, -1].argmax(dim=-1)
generated = tok.decode(next_ids)

top5 = acts["mlp_out"].topk(5).indices.tolist()
print(f"Prompt  : {prompt}")
print(f"Response: {generated}")
print(f"Top-5 neuron IDs in last layer: {top5}")
