<div align="center">
<h1>Beyond Artificial Misalignment: Detecting and Grounding Semantic-Coordinated Multimodal Manipulation(ACM MM2025)</h1>

<div>
  Jinjie Shen<sup>1</sup></a>
  Yaxiong Wang<sup>1</sup></a>
  Lechao Chen<sup>1</sup></a>
  Pu Nan<sup>2</sup></a>
  Zhun Zhong<sup>1</sup></a>
</div>
</div>

## News
- [07/2025] RamDG Code is released.

## Introduction
This is the official implementation of *SAMM* and *RamDG*. We propose a realistic research scenario: detecting and grounding semantic-coordinated multimodal manipulations, and introduce a new dataset SAMM. To address this challenge, we design the RamDG framework, proposing a novel approach for detecting fake news by leveraging external knowledge.

The framework of the proposed RamDG:





## 🔧 Dependencies and Installation
### Download
```
mkdir code
cd code
git clone https://github.com/shen8424/SAMM-RamDG-CAP.git
cd SAMM-RamDG-CAP
```

### Environment
```
conda create -n RamDG python=3.8
conda activate RamDG
conda install --yes -c pytorch pytorch=1.10.0 torchvision==0.11.1 cudatoolkit=11.3
pip install -r requirements.txt
conda install -c conda-forge ruamel_yaml
```

### ⏬ Prepare Checkpoint
Download the pre-trained model through this link: [ALBEF_4M.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth) and [pytorch_model.bin](https://drive.google.com/file/d/15qfsTHPB-CkEVreOyf-056JWDAVjWK3w/view?usp=sharing)[GoogleDrive].

Then put the `ALBEF_4M.pth` and `pytorch_model.bin` into `./code/SAMM-RamDG-CAP/`.

```
./
├── code
    └── SAMM-RamDG-CAP (this github repo)
        ├── configs
        │   └──...
        ├── dataset
        │   └──...
        ├── models
        │   └──...
        ...
        └── ALBEF_4M.pth
        └── pytorch_model.bin
```

## ⏬ Prepare Data
### Brief introduction

We present <b>SAMM</b>, a large-scale dataset for Detecting and Grounding Semantic-Coordinated Multimodal Manipulation.



