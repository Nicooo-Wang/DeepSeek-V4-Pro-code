# Inference code for DeepSeek models

First convert huggingface model weight files to the format of this project.
```bash
export EXPERTS=384
export MP=8
export CONFIG=config.json
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Then chat with DeepSeek model at will!
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

Or batch inference from file.
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

Or multi nodes inference.
```bash
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

If you want to use fp8, just remove `"expert_dtype": "fp4"` in `config.json` and specify `--expert-dtype fp8` in `convert.py`.

## Run `model.py` without weights (structure learning / debugging)

`model.py` has a `__main__` smoke test that builds the **real DeepSeek-V4-Pro architecture**
and runs a prefill + decode + MTP pass — **no checkpoint required** (random weights). It uses
the pure-PyTorch reference kernels in `kernel.py` (the original tilelang/CUDA kernels are kept
in `kernel_tilelang.py`), so every operator is single-step debuggable. Supports **1 GPU or
2 GPU tensor-parallel**.

> The `ModelArgs` defaults equal the real `config.json`; only `n_layers` is reduced
> (default 2; real 61) for memory. Override the layer count without editing code via the
> `MODEL_N_LAYERS` env var.

### One-time setup (uv venv)

Run from the repo root. Pin `torch` to a build matching your NVIDIA driver — for driver 12.8
use the `cu128` index (the default `cu128` wheel is now `cu130` and will report "driver too old"):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python numpy
```

### Run

```bash
cd inference

# 1 GPU (pick a GPU with enough free memory, e.g. GPU 7)
CUDA_VISIBLE_DEVICES=7 ../.venv/bin/python model.py

# 2 GPUs, tensor parallel (pick two free GPUs; add --master-port=NNNN if 29500 is busy)
CUDA_VISIBLE_DEVICES=6,7 ../.venv/bin/torchrun --nproc-per-node 2 model.py

# Add layers to see more structure without editing code (each layer ≈ 14 GB):
CUDA_VISIBLE_DEVICES=7 MODEL_N_LAYERS=4 ../.venv/bin/python model.py
```

Expected output (1 GPU, `n_layers=2`): prefill / decode / MTP each print `(2, 129280)`, plus a
per-rank peak-memory line. Under TP=2 the per-GPU parameter count and memory both halve.

### Memory

Each MoE layer (384 fp4 experts) needs ≈ 14 GB, so `n_layers=2` ≈ 47 GB. Tensor parallel halves
the per-GPU cost (≈ 25 GB / rank at `n_layers=2`). If you hit OOM, lower `MODEL_N_LAYERS`. The
launcher sets `expandable_segments:True` to reduce fragmentation. (`fp4` expert weights are
unpacked by a manual nibble decoder in `kernel.py`, since torch 2.8 cannot cast
`float4_e2m1fn_x2` to float directly.)

### What to step through (all in `inference/`)

- `Transformer.forward` / `Block.forward` — overall flow + Hyper-Connection residuals.
- `kernel.py: sparse_attn` — sparse MLA attention; `hc_split_sinkhorn` — HC mixing.
- `Compressor` / `Indexer` — KV compression + top-k position selection.
- `MoE` + `Gate` — expert routing (hash routing for the first `n_hash_layers` layers,
  score-based top-k afterwards).
- TP sharding: `ColumnParallelLinear` / `RowParallelLinear` (output / input dim + `all_reduce`)
  and `ParallelEmbedding` / `ParallelHead` (vocab dim + `all_gather`).
