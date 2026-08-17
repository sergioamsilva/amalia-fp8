"""Quantize AMALIA-9B to FP8 (W8A8 dynamic) with llm-compressor.

Runs inside the vLLM container, which already ships torch and transformers built for
aarch64 / CUDA 13.

Why FP8_DYNAMIC:
  - weights in FP8 per channel, activations in FP8 computed at runtime
  - requires NO calibration dataset, so there is no risk of degrading European Portuguese
    by calibrating on English text (the default in every pipeline)
  - compressed-tensors format, which vLLM loads natively
  - FP8 is native on Blackwell tensor cores (sm_121)

lm_head is left at original precision.
"""
import os, time

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SOURCE = os.environ.get("SOURCE", "amalia-llm/AMALIA-9B-0626-DPO")
DEST = os.environ.get("DEST", "/root/.cache/huggingface/AMALIA-9B-0626-DPO-FP8")

print(f"source: {SOURCE}")
print(f"dest:   {DEST}")

t0 = time.time()
print("loading model in bf16...")
model = AutoModelForCausalLM.from_pretrained(SOURCE, dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(SOURCE)
print(f"  loaded in {time.time()-t0:.0f}s")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

print("quantizing to FP8_DYNAMIC (no calibration)...")
t1 = time.time()
oneshot(model=model, recipe=recipe)
print(f"  quantized in {time.time()-t1:.0f}s")

print("saving...")
t2 = time.time()
model.save_pretrained(DEST, save_compressed=True)
tok.save_pretrained(DEST)
print(f"  saved in {time.time()-t2:.0f}s")
print(f"TOTAL: {time.time()-t0:.0f}s")
