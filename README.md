# Unrolled-ADMM — Lensless Computational Imaging

Realization of Unrolled-ADMM within DL-research course. This work is based on:

- [Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502)
- [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102)

## Methods from papers that are implemented

| Method | Config | Checkpoint in Google drive |
|-|-|-|
| ADMM-100 (100 fixed iterations) | `model=admm100` | Not required |
| Unrolled ADMM-20 | `model=admm20` | `checkpoints/admm20/model_best.pth` |
| Modular LeADMM-5 (pre + post) | `model=modular_leadmm5`, `model.variant=pre_post` | `checkpoints/leadmm5-pre_post/model_best.pth` |
| Modular LeADMM-5 (pre only) | `model=modular_leadmm5`, `model.variant=pre` | `checkpoints/leadmm5-pre/model_best.pth` |
| Modular LeADMM-5 (post only) | `model=modular_leadmm5`, `model.variant=post` | `checkpoints/leadmm5-post/model_best.pth` |

## First steps

Run these commands to set your Comet-ml API key, clone the repository and install dependencies:

```bash
export COMET_API_KEY=your_key

git clone https://github.com/Vdv09/Unrolled-ADMM.git
cd Unrolled-ADMM

python -m venv env
source env/bin/activate

pip install -r requirements.txt
pip install scipy pyffs opencv-python gdown
```

## Download checkpoints

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1jhGziJtGEc1pitPdfIyvdhp_Xt9NuH2C" -O checkpoints --remaining-ok
```

## Training

Here desribed commands to train different models

**Unrolled ADMM-20:**

```bash
python train.py --config-name train_admm20 writer.run_name=admm20
```

**Modular LeADMM-5** 

Here you must set `model.variant` to `pre`, `post`, or `pre_post` - from it depends will you use preprocessor or/and postprocessor

```bash
python train.py --config-name train_leadmm5 writer.run_name=leadmm5-pre_post model.variant=pre_post
```

## Inference

Outputs come to: `data/saved/<save_path>/inference/{image_id}.png`

**Example**:

```bash
python inference.py \
  datasets=digicam_eval \
  model=modular_leadmm5 \
  model.variant=pre_post \
  inferencer.from_pretrained=checkpoints/leadmm5-pre_post/model_best.pth \
  inferencer.save_path=modular_leadmm5_inference
```

## Metrics

```bash
python calculate_metrics.py \
  --pred-dir data/saved/modular_leadmm5_inference/inference \
  --dataset-dir /path/to/data
```

## Speed benchmark

Single model:

```bash
python scripts/check_speed.py --model admm100 --device cuda
python scripts/check_speed.py --model admm20 --checkpoint checkpoints/admm20/model_best.pth --device cuda
```

## Demo

You can see also demo.ipynb in this repo to deeply understand project structure