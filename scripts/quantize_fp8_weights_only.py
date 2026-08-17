"""Quantize AMALIA-9B to FP8 WEIGHTS ONLY (W8A16), leaving activations in BF16.

Difference from quantize_fp8.py, which uses FP8_DYNAMIC (W8A8):

  FP8_DYNAMIC   weights in FP8 per channel + activations in FP8 computed at runtime
  this script   weights in FP8 per channel + activations untouched in BF16

The hypothesis this was written to test: quantization error should concentrate in the
activations, which vary per token and have outliers, while weights are static and well
behaved. On a memory-bandwidth-bound machine like the DGX Spark (GB10, ~273 GB/s) the
speedup comes almost entirely from reading half the weight bytes, not from faster
arithmetic. So weight-only quantization should keep nearly all the speed with a fraction
of the error.

It did not. Weight-only measured +2.31% perplexity against +0.94% for W8A8, while being
only marginally faster. See the README: the weights are quantized identically in both
variants, so the whole difference comes from the compute path, and the native FP8 tensor
core path turned out to be the more accurate one.

This script is kept so the result can be reproduced and challenged.
"""
import os, time

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
)

SOURCE = os.environ.get("SOURCE", "amalia-llm/AMALIA-9B-0626-DPO")
DEST = os.environ.get("DEST", "/root/.cache/huggingface/AMALIA-9B-0626-DPO-FP8-WEIGHTS")

print(f"source: {SOURCE}")
print(f"dest:   {DEST}")

t0 = time.time()
print("loading model in bf16...")
model = AutoModelForCausalLM.from_pretrained(SOURCE, dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(SOURCE)
print(f"  loaded in {time.time()-t0:.0f}s")

scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=None,   # <- the only difference: activations stay in BF16
    output_activations=None,
)

recipe = QuantizationModifier(
    config_groups={"weights_fp8": scheme},
    ignore=["lm_head"],
)

print("quantizing weights only (W8A16), no calibration...")
t1 = time.time()
oneshot(model=model, recipe=recipe)
print(f"  quantized in {time.time()-t1:.0f}s")

print("saving...")
t2 = time.time()
model.save_pretrained(DEST, save_compressed=True)
tok.save_pretrained(DEST)
print(f"  saved in {time.time()-t2:.0f}s")
print(f"TOTAL: {time.time()-t0:.0f}s")
