# EAQuant

EAQuant adds evidence-aware token KL and answer-recovery KL objectives to
post-training quantization. The retained implementation supports OSTQuant and
OmniQuant.

## Setup

Use separate environments for the two backends. Install a CUDA-compatible
PyTorch build first, then the recorded dependencies:

```bash
# OSTQuant: PyTorch 2.2.1
python -m pip install -r backends/ostquant/requirements.txt
python -m pip install fast-hadamard-transform==1.0.4.post1 --no-build-isolation
python -m pip install -e . --no-deps
```

```bash
# OmniQuant: PyTorch 2.4.1 and torchvision 0.19.1
python -m pip install -r backends/omniquant/requirements.txt
python -m pip install -e . --no-deps
```

The paper experiments used NVIDIA L40S GPUs: four GPUs for OSTQuant
quantization and one GPU for OmniQuant quantization and inference.

## Data and models

Download [MedExpQA](https://github.com/hitz-zentroa/MedExpQA),
[MedExQA](https://github.com/knowlab/MedExQA), and
[ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA).
Place datasets and Hugging Face models under `data/`:

```text
data/
├── med_datasets/
│   ├── MedExpQA/data/en/
│   ├── MedExQA/dev/
│   └── challengeclinicalQA/
└── models/
    ├── Mistral-7B-Instruct-v0.3/
    ├── Meta-Llama-3-8B-Instruct/
    ├── BioMistral-7B/
    ├── Llama3-OpenBioLLM-8B/
    └── Qwen2.5-72B-Instruct/
```

All generated caches, checkpoints, predictions, and metrics are written under
`outputs/` automatically. Both `data/` and `outputs/` are excluded from Git.

## Reproduction

Run commands from the repository root. `scripts/run_paper.py` contains the
retained paper settings and creates the required output directories. Every
command below explicitly identifies the model; quantized stages additionally
identify the backend and seed.

| `--model` value | Model | OSTQuant | OmniQuant |
| --- | --- | --- | --- |
| `mistral_7b_instruct` | Mistral-7B-Instruct-v0.3 | Yes | Yes |
| `llama3_8b_instruct` | Meta-Llama-3-8B-Instruct | Yes | Yes |
| `biomistral_7b` | BioMistral-7B | Yes | — |
| `openbiollm_8b` | Llama3-OpenBioLLM-8B | Yes | — |

### 1. Evidence cache

The evidence cache is model-specific but shared across quantization seeds:

```bash
python scripts/run_paper.py cache --model mistral_7b_instruct
```

This command internally creates the FP16 teacher cache, the standard seed-0
checkpoint needed to measure quantization damage, Qwen evidence, and the paper
filter. BioMistral and OpenBioLLM automatically omit the final filter.

### 2. Quantization

Run all backend-specific quantization steps with one command:

```bash
python scripts/run_paper.py quantize --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py quantize --model mistral_7b_instruct --backend omniquant --seed 0
```

Each command produces both the standard MedMix baseline and EAQuant artifacts.
The OmniQuant command handles activation statistics and its seed-matched effect
cache internally. Completed prerequisites are reused; add `--force` only to
rebuild them. Repeat with `--seed 0`, `--seed 1`, and `--seed 2`.

### 3. Inference

```bash
python scripts/run_paper.py infer-fp16 --model mistral_7b_instruct
python scripts/run_paper.py infer-baseline --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py infer-eaquant --model mistral_7b_instruct --backend ostquant --seed 0
```

For OmniQuant, replace `--backend ostquant` with `--backend omniquant`. The
FP16 prediction is shared across backends and seeds.

### 4. Evaluation

Every metric consumes the same three inference outputs: FP16, standard MedMix,
and EAQuant.

| Stage | Metric | OpenAI API |
| --- | --- | --- |
| `accuracy` | Answer accuracy for all three systems | No |
| `agreement` | Baseline/EAQuant answer agreement with FP16 | No |
| `fec` | FP16 Evidence Coverage (evidence retention) | GPT-5.4 |
| `uacr` | Unsupported Added Claim Rate | GPT-5.4 |
| `pairwise` | Blind standard-MedMix versus EAQuant LLM judge | GPT-5.4 |

```bash
python scripts/run_paper.py accuracy --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py agreement --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py fec --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py uacr --model mistral_7b_instruct --backend ostquant --seed 0
python scripts/run_paper.py pairwise --model mistral_7b_instruct --backend ostquant --seed 0
```

Set `OPENAI_API_KEY` before running `fec`, `uacr`, or `pairwise`. The wrapper
automatically resolves and passes the matching three prediction JSONL files;
it prints their paths and stops with a missing-file error if inference has not
finished. Use the same `--model`, `--backend`, and `--seed` values for inference
and evaluation.

### 5. Ablations

Build the rationale and cache-stage variants once:

```bash
python scripts/run_paper.py rationale-caches --model mistral_7b_instruct --seed 0
python scripts/run_paper.py cache-stage-caches --model mistral_7b_instruct
```

Train and infer a selected variant with the same one-line interface:

```bash
python scripts/run_paper.py ablation-train --model mistral_7b_instruct --seed 0 --variant token
python scripts/run_paper.py ablation-infer --model mistral_7b_instruct --seed 0 --variant token
```

Available variants are `no-ea`, `token`, `recovery`, `both`, `full`, `random`,
and `cache-v1` through `cache-v4`. After producing the `full` and `random`
predictions, run their shared-cohort FEC comparison with:

```bash
python scripts/run_paper.py rationale-eval --model mistral_7b_instruct --seed 0
```

Use the lower-level scripts directly only when intentionally changing paper
hyperparameters.

## License

New EAQuant code is released under the root MIT license. The OmniQuant and
OSTQuant backend code retains its upstream licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
