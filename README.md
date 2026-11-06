# PairAlign

Code for training and evaluating the PairAlign audio sequence tokenizer.

## Setup

The project is pinned by `pyproject.toml` and `uv.lock`. A local virtual environment is intentionally not included in the repository.

Use Python 3.10--3.12 for the pinned PyTorch/Torchaudio 2.5.1 environment.

```bash
uv sync --frozen
```

Update the dataset and checkpoint paths in `config/mainFull.yaml` before running training.

## Training

The root entry point can select the training stage directly:

```bash
python main.py --config mainFull.yaml --model AudioTokenizer --ckpt_dir stage1
python main.py --config mainFull.yaml --model AudioTokenizerARPrior --ckpt_dir stage2
python main.py --config mainFull.yaml --model AudioTokenizerAR_FINAL --ckpt_dir pairalign
```

For Stage II, set `checkpoints.stage1` to the Stage-I checkpoint. For Stage III, set `checkpoints.stage2` (or `checkpoints.ar_prior`) to the Stage-II checkpoint.

The paired dataset always uses a 3-second context. 

## Evaluation scripts

Scripts under `src/` expose their paths and checkpoints as command-line arguments. Run a script with `--help` for its required inputs. 
