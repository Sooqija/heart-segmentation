# Whole Heart Segmentation

## Task



![alt text](https://zmiclab.github.io/zxh/0/mmwhs/res/WholeHeartSegment_ErrorMap_WhiteBg.gif)


## Installation

To install console application `hsa` - heart segmentation app, run the following command.

```
pip install .
```

To see all options run `hsa` with `--help` flag.

```
hsa --help
```

## Examples

Example command to run script to train a model

```
hsa train --model unetr --image_dir "./image_folder" --label_dir "./data/label_folder" --output_dir "./runs" --tag "your-run-tag" --epochs 250 --seed 0 --dataset_config "./data/data_folder/dataset.json"
```




## Datasets
### MM-WHS 2017 Dataset

The MM-WHS 2017 dataset is a dataset for multi-modality whole heart segmentation. It provides 20 labeled and 40 unlabeled CT volumes, as well as 20 labeled and 40 unlabeled MR volumes. In total there are 120 multi-modality cardiac images acquired in a real clinical environment.
**Publication Date: 2017**.

#### Sources

Official Website: https://zmiclab.github.io/zxh/0/mmwhs/

Download Link: https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA

More About Dataset: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MM-WHS.md

![alt text](https://zmiclab.github.io/zxh/0/mmwhs/res/MMData2.png)
*Official visualization*

![alt text](https://github.com/Sooqija/heart-segmentation/blob/main/figures/vtk.png)
*Visualization via VTK*

### WHS CHD MICCAI19 Dataset


**Publication Date: 2023**.

#### Sources

Original Paper: 

Download Link: https://www.kaggle.com/datasets/xiaoweixumedicalai/chd68-segmentation-dataset-miccai19
