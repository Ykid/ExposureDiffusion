
# [ICCV 2023] ExposureDiffusion: Learning to Expose for Low-light Image Enhancement
Welcome! This is the official implementation of the paper "[ExposureDiffusion: Learning to Expose for Low-light Image Enhancement](https://arxiv.org/pdf/2307.07710.pdf)".

- *Yufei Wang, Yi Yu, Wenhan Yang, Lanqing Guo, Lap-Pui Chau, Alex C. Kot, Bihan Wen*

- $^1$ Nanyang Technological University 
$^2$ Peng Cheng Laboratory
$^3$ The Hong Kong Polytechnic University


## Prerequisites
Besides, you may need to download the ELD dataset and SID dataset as follows
- SID ([official project](https://github.com/cchen156/Learning-to-See-in-the-Dark)):  [download (25 GB)](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)


## Quick setup (local notes)
- Environment: CUDA compilation tools 11.8.89, Python 3.6.15.
- Upgrade tooling then install deps:
  ```bash
  python -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"
  pip install -r requirements.txt
  ```
- Datasets: download SID and place under `datasets/SID/`.
- Add the following to your `.env`:
  ```bash
  WANDB_API_KEY=<xxxx>
  CUDA_VISIBLE_DEVICES=0
  ED_LMDB_CPU_COUNT=1
  ED_PREFETCH_FACTOR=2
  ED_SAVE_EPOCH_FREQ=1
  ED_EVAL_PER_N_EPOCH=1
  ED_ENABLE_DEBUG=0
  ```
- Evaluation: run `bash run_inference.sh`.
- Training: run `bash run_training.sh`.




## Pre-trained models
You can download the pre-trained models from [google drive](https://drive.google.com/drive/folders/1qdxZBg-GsYxHaDj3Zd_NikYiweqDOFzo?usp=drive_link), which includes the following models 
- The UNet model trained on P+g noise model (```SID_Pg.pt```), ELD noise model (```SID_PGru.pt```), and real-captured paired dataset (```SID_real.pt```). 
- The NAFNet model trained on P+g (```SID_pg_naf2.pt```) and ELD noise models (```SID_PGru_naf2.pt```).



## Citation
If you find our code helpful in your research or work please cite our paper.

```bibtex
@article{wang2023exposurediffusion,
  title={ExposureDiffusion: Learning to Expose for Low-light Image Enhancement},
  author={Wang, Yufei and Yu, Yi and Yang, Wenhan and Guo, Lanqing and Chau, Lap-Pui and Kot, Alex C and Wen, Bihan},
  journal={arXiv preprint arXiv:2307.07710},
  year={2023}
}
```
## Copyright
The purpose of the use is non-commercial research and/or private study.

## Acknowledgement
This work is based on [ELD](https://github.com/Vandermode/ELD) and [PMN](https://github.com/megvii-research/PMN.). We sincelely appreciate the support from the authors.

## Contact
If you would like to get in-depth help from me, please feel free to contact me (yufei001@ntu.edu.sg) with a brief self-introduction (including your name, affiliation, and position).
