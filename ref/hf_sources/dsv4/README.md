# Minimal inference

This directory contains a readable reference implementation rather than a
production serving engine. The model code includes Vision + Aligner, DFlash,
MoE, Hyper-Connections, and `Transformer.forward_spec()` for the DSpark forward
path. The generation loop remains straightforward autoregressive sampling.

## Install

```bash
python -m pip install -r requirements.txt
```

## Convert Hugging Face weights

The reference runtime uses one converted checkpoint file per tensor-parallel
rank. From this directory:

```bash
export HF_CKPT_PATH=/path/to/DeepSeek-V4-Flash-Vision-Exp-HF
export SAVE_PATH=/path/to/DeepSeek-V4-Flash-Vision-Exp-TP4
export MP=4

python convert.py \
  --hf-ckpt-path "${HF_CKPT_PATH}" \
  --save-path "${SAVE_PATH}" \
  --n-experts 256 \
  --model-parallel "${MP}" \
  --expert-dtype fp4
```

`convert.py` also copies `tokenizer.json` and `tokenizer_config.json` into the
converted checkpoint directory. `--tokenizer-path` can be used when tokenizer
files live outside the weight directory.

## Run the equivalent TXT and JSON examples

```bash
export CKPT_PATH=/path/to/DeepSeek-V4-Flash-Vision-Exp-TP4
export MP=4

INPUT_FILE=examples/example_vl.txt ./run.sh
INPUT_FILE=examples/example_vl_harmony.json ./run.sh
```

The two files express the same interleaved two-image prompt and therefore
produce identical encoded prompts and input token IDs.

For interactive chat:

```bash
torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${CKPT_PATH}" \
  --config config.json \
  --interactive \
  --temperature 1.0
```

For multi-node execution, pass the usual `torchrun --nnodes`, `--node-rank`,
`--master-addr`, and `--master-port` arguments before `generate.py`.

## Preprocessing tests

From the repository root:

```bash
python -m pytest -q \
  encoding/test_encoding_dsv4.py \
  inference/test_image_processor.py
```
