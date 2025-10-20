# SR-ICL

Unified Medical Lesion Segmentation via Self-referring Indicator

### Datasets

Wet AMD: [AMD-SD](https://www.kaggle.com/datasets/gaoweihao/amd-sd)
Brain Tumor: [BTD](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)


In the original implementation (dataset.py), you need to split each dataset into training and validation sets, and then write the lists of images for the training and validation sets into separate text files.

Here is a better implementation (dataset_new.py): you only need to provide the dataset path, and it will automatically split the dataset into a 4-fold for cross-validation.
