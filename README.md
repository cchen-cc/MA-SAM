# MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation

This is a PyTorch implementation of the paper [MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation](https://arxiv.org).

![Overview of MA-SAM framework](asset/overview.png?raw=true "Overview of MA-SAM framework")

>  We introduce a modality-agnostic SAM adaptation framework, named as MA-SAM, that is applicable to various volumetric and video medical data. Our method has been comprehensively evaluated on four medical image segmentation tasks, by using 10 public datasets across CT, MRI, and surgical video data. Without using any prompt, our method consistently outperforms various state-of-the-art 3D approaches, surpassing nnU-Net by 0.9%, 2.6%, and 9.9% in Dice for CT multi-organ segmentation, MRI prostate segmentation, and surgical scene segmentation respectively. Our model also demonstrates strong generalization, and excels in challenging tumor segmentation when prompts are used.

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
- BTCV dataset: The raw data can be downloaded from the [challenge website](https://www.synapse.org/#!Synapse:syn3379050) after registration. We also provided our [preprocessing script](https://github.com/cchen-cc/MA-SAM/blob/main/preprocessing/util_script_btcv.py) and preprocessed data. 
- Prostate dataset: The raw data can be downloaded from [this link](https://liuquande.github.io/SAML/). 
- EndoVis'18 dataset: The raw data can be downloaded form the [challenge website](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) after registration.
- MSD-Pancreas: The raw data can be downloaded from the [this link](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).

In the directory [preprocessing](), we provide the dataset splits that are used in our work. 
  
## Training
To train MA-SAM, first go to the folder MA-SAM, then start the training:
```sh
cd MA-SAM
python train.py --root_path <Your data directory> --output <Your output directory> --ckpt <Your SAM pre-trained model directory>
```
We use 8 A100 80G GPUs to train our full model. To reduce the memory consumption, you may consider change the backbone from ViT_H to ViT_L or ViT_B. To do so, you would need to change the arguments --vit_name to 'vit_l' or 'vit_b' and load the correct SAM pre-trained weights for --ckpt. You may also consider reduce the number of consecutive slices. To do so, you would need to make according changes for the data pre-processing and evaluation. However, using smaller backbone or reduce the number of consecutive slices would lead to a decrease in performance. We do not recommend to reduce the batch size, which would make the model difficult to converge.

## Inference
We provide [our trained model](https://drive.google.com/file/d/1zBaDHkkH9FbPC2S8vl6cwUqy5nrxPmtu/view?usp=drive_link) for reproducing our results. 
To perform inference with the trained MA-SAM model, use the following command
```sh
python test.py --adapt_ckpt <Your MA-SAM model directory> --data_path <Your data directory> --ckpt <Your SAM pre-trained model directory> --is_savenii
```
Running this command will output the Dice evaluation metrics for your model. The argument --is_savenii will create a folder with the same name as your MA-SAM model directory (without the .pth postfix of course) to save the corresponding .nii prediction files.

## Acknowledgments
Our code is based on [SAMed](https://github.com/hitachinsk/SAMed) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We appreciate the authors for their great works. 

## Citation
If you find the code useful for your research, please cite our paper.
```sh
```

## Note
- The repository is being updated. Please feel free to contact us or open new issues if you encounter any problem when using our code.
- Contact: Cheng Chen ([cchen101@mgh.harvard.edu]())
