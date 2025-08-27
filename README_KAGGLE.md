# Running this project on Kaggle

This repository has been adapted to run seamlessly in Kaggle notebooks with
optional multi‑GPU support and automated persistence of results.  Follow the
steps below to reproduce your experiments without any interactive prompts.

## Setup

1. Open the repository as a Kaggle notebook (use the “Add Notebook” button on
   Kaggle and select this repo).
2. In the notebook settings, enable **GPU** and choose **2** GPUs if
   available.
3. If you would like your artefacts to be uploaded as a private Kaggle
   dataset, add your Kaggle API credentials in the notebook under
   *Add-ons → Secrets* with keys `KAGGLE_USERNAME` and `KAGGLE_KEY`.  These
   credentials are optional; when absent the artefacts will still persist
   through the notebook outputs.

## Launch training

Use the provided launch script to automatically detect the number of GPUs and
start training:

```bash
!bash scripts/launch_pytorch.sh --models dcgan --datasets mnist --epochs 25
```

The script will call `torchrun` with one process per GPU when multiple GPUs
are available.  For single‑GPU or CPU runs it falls back to a normal Python
invocation.  You can pass any of the usual `main.py` arguments after the
script invocation.  For example, to train multiple models:

```bash
!bash scripts/launch_pytorch.sh --models vanilla dcgan --datasets mnist cifar10 --epochs 50
```

## Persistence of results

All models, checkpoints, samples, metrics and plots are written to the
`outputs/` directory.  On Kaggle this directory maps to
`/kaggle/working/outputs` which is automatically persisted as a notebook
output when you **Save & Submit** the notebook.  If Kaggle API credentials
are provided, running

```bash
python persist/kaggle_dataset_push.py
```

will version your outputs as a private dataset under your Kaggle account.  This
script is non‑interactive and will quietly skip the upload if credentials are
missing.

## Notes

* The training loop has been updated to use distributed data parallelism
  (DDP) when more than one GPU is available.  The dataset loaders
  automatically employ `DistributedSampler` in this case.
* When training in distributed mode only the rank‑0 process saves checkpoints,
  samples and plots.
* You can customise hyper‑parameters for your Kaggle runs by editing the file
  `configs/kaggle.yaml` or by passing arguments on the command line.