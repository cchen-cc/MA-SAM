# MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation

This is a PyTorch implementation of the paper [MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation](https://arxiv.org).

![Overview of MA-SAM framework](asset/overview.png?raw=true "Overview of MA-SAM framework")

>  We introduce a modality-agnostic SAM adaptation framework, named as MA-SAM, that is applicable to various volumetric and video medical data. Our method roots in the parameter-efficient fine-tuning strategy to update only a small portion of weight increments while preserving the majority of SAM’s pre-trained weights. By injecting a series of 3D adapters into the transformer blocks of the image encoder, our method enables the pre-trained 2D backbone to extract third-dimensional information from input data. The effectiveness of our method has been comprehensively evaluated on four medical image segmentation tasks, by using 10 public datasets across CT, MRI, and surgical video data. Remarkably, without using any prompt, our method consistently outperforms various state-of-the-art 3D approaches, surpassing nnU-Net by 0.9%, 2.6%, and 9.9% in Dice for CT multi-organ segmentation, MRI prostate segmentation, and surgical scene segmentation respectively. Our model also demonstrates strong generalization, and excels in challenging tumor segmentation when prompts are used.

## Usage
#### Environmental Requirements
- Ubuntu 20.04
- Anaconda
- Python=3.10.12
- torch==2.0.1
- cuda==11.7

#### Installation
Clone this repository and then install the dependencies.
```sh
git clone https://github.com/cchen-cc/MA-SAM.git
conda create -n masam python=3.10.12
conda activate masam
pip install -r requirements.txt
```

## Data Preparation
- BTCV dataset: The raw data can be downloaded from the [challenge website](https://www.synapse.org/#!Synapse:syn3379050) after registration. We also provided our preprocessing script and preprocessed data. 
- Prostate dataset: The raw data can be downloaded from [this link](https://liuquande.github.io/SAML/). 
- EndoVis'18 dataset: The raw data can be downloaded form the [challenge website](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) after registration.
- MSD-Pancreas: The raw data can be downloaded from the [this link](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). 
  
## Training
