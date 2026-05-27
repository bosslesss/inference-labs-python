"""Inference Labs quickstart.

Set INFERENCE_LABS_API_KEY in your environment, then:

    python examples/quickstart.py
"""
from inference_labs import InferenceLabs

client = InferenceLabs()

# 1) Simplest possible call -- router picks the model.
out = client.generate(prompt="Write one sentence about vendor-neutral AI routing.")
print(f"[{out.provider}/{out.model}] {out.text}")
print(f"  cost ${out.cost_usd:.5f}  latency {out.latency_ms} ms  cached={out.cached}")

# 2) Policy-driven: cheapest model that meets a quality bar.
out = client.generate(
    prompt="Summarize: 'The fox jumped over the lazy dog'.",
    strategy="cost-first",
    max_cost_usd=0.001,
)
print(f"\n[cost-first] {out.text}")

# 3) Constrain to a specific model allowlist.
out = client.generate(
    prompt="Reply with exactly the word OK.",
    allow_models=["gpt-4o-mini", "claude-haiku-4-5"],
)
print(f"\n[allowlist -> {out.model}] {out.text}")

# 4) Streaming
print("\n[streaming] ", end="", flush=True)
for chunk in client.stream(prompt="Count from 1 to 5, one number per line."):
    print(chunk, end="", flush=True)
print()
