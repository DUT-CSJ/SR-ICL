# SR-ICL

Code of Unified Medical Lesion Segmentation via Self-referring Indicator

### Environment

```
conda create -n sricl python=3.8
conda activate sricl
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install timm==1.0.9
# Install any other dependencies as needed.
```

### Datasets

Wet AMD: [AMD-SD](https://www.kaggle.com/datasets/gaoweihao/amd-sd)

Brain Tumor: [BTD](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

Adenocarcinoma: [EBHI-Seg](https://www.kaggle.com/datasets/orvile/ebhi-seg-colorectal-cancer)

Thyroid Nodule: [TNUI 2021](https://github.com/zxg3017/TNUI-2021-)

Colon Polyp: [Combined datasets](https://drive.google.com/file/d/1A29IkVysVPUPy4vu1RklKf4AAD7QvV3x/view?usp=sharing)

Lung Infection: [Dataset1](https://medicalsegmentation.com/COVID19/) and [Dataset2](https://cir.nii.ac.jp/crid/1882553967772574976)

Breast Lesion: [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

Skin Lesion: [ISIC 2018](https://challenge.isic-archive.com/data/#2018)

You can add any other binary segmentation datasets.

### Training and Inference

Training

```
sh run.sh
```

Inference

```
python inference.py
```

### Note

In the original implementation (dataset.py), you need to split each dataset into training and validation sets, and then write the lists of images for the training and validation sets into separate text files.

Here is a better implementation (dataset_new.py): you only need to provide the dataset path, and it will automatically split the dataset into a 4-fold for cross-validation.

### Cite

```
@InProceedings{sricl,
    author    = {Chang, Shijie and Zhao, Xiaoqi and Zhang, Lihe and Wang, Tiancheng},
    title     = {Unified Medical Lesion Segmentation via Self-referring Indicator},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10414-10424}
}
```
