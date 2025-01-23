# [Unleashing the Power of Generic Segmentation Models: A Simple Baseline for Infrared Small Target Detection]

[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2409.04714) 

This repository contains the official implementation of the paper:

**[Unleashing the Power of Generic Segmentation Models: A Simple Baseline for Infrared Small Target Detection]**
> Published in ACM MM 2024

**Untested version, still cleaning my code!**
## Abstract

(A simple effective IRSTD model.)

---

## Getting Started

### Installation
   
1. Clone the repository:

   ```bash
   git clone https://github.com/O937-blip/SimIR.git
   cd SimIR
   git submodule update --init --recursive
   ```

2. Set up a virtual environment and install dependencies:

   ```pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
   python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
   pip install git+https://github.com/cocodataset/panopticapi.git
   git clone https://github.com/UX-Decoder/Semantic-SAM
   cd Semantic-SAM
   python -m pip install -r requirements.txt
   ```

3. Copy files into the corresponding folder in ./Semantic-SAM and modify __init__.py to ensure the files are loadable.



### Dataset Preparation

1. Please follow [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM/blob/main/DATASET.md) to prepare your data, tsv format is recommended. 
2. Export your datapath
   ```
   export IRDST_DATASETS=''
   export MDFA_DATASETS=''
   export IRSTD_DATASETS=''
   export NUAA_DATASETS=''
   export NUDT_DATASETS=''
   ```
3. The dataloader follows Detectron2 that contains:
   (1) A dataset registrator
   (2) A dataset mapper.
   Here we provide a mapper and a registrator template for the NUDT dataset, tsv format. If you want to load your datasets in other formats, please read the tutorial of [detectron 2](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html).
