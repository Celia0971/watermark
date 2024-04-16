# Primary Key Free Watermarking

## Environment
- The code is developed using python 3.9 on Ubuntu 20.04.5 LTS. 
- Install dependencies: `pip install -r requirements.txt`

## Watermark-Removing Attacks
- Run scripts in ~/attack_channel/*.py to attack.
- Details of attack related parameters can be found at args_parse, --help in each ~/attack_channel/*.py file.

## Watermark Embedding
- Run `python ./embed.py` to embed the watermark into dataset. 
- The user has the flexibility to customize various parameters, including the number of columns in which to embed the watermark, the secret key, and the number of watermarked datasets, etc. Details of related settings and parameters can be found at args_parse, --help in embed.py.
- Watermarked datasets are saved by default in the folder `./experiments/exp_tag/dataset/embed/wm_datasets`
- Secret keys are saved by default in the folder `./experiments/exp_tag/dataset/embed/storedkeys`

## Watermark Detection
- Run `python ./detect.py` to detect the watermark from a suspicious dataset. 
- Details of related settings and parameters can be found at args_parse, --help in detect.py.
- Watermark detection results are saved by default in the folder `./experiments/exp_tag/dataset/extract`

## Quick Start
- Run `bash ./fct_robust_ans_bash_run.sh` to conduct watermark embedding, watermark-removing attacks and watermark detection for dataset FCT.
- Run `bash ./gsad_robust_ans_bash_run.sh` to conduct watermark embedding, watermark-removing attacks and watermark detection for dataset GSAD.

