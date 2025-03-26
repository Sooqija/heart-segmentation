# Whole Heart Segmentation

## Task

The goal of whole heart segmentation is to extract the volume and shapes of all the substructures of the heart, commonly including the blood cavities of the four chambers, left ventricle myocardium, and sometimes the great vessels as well if they are of interest. The blood cavities of the fourchambers are: the left ventricle blood cavity, the right ventricle blood cavity, the left atrium blood cavity and the right atrium blood cavity.

All goal cardiac substructures:
- Left ventricle (LV)
- Right ventricle (RV)
- Left atrium (LA)
- Right atrium (RA)
- Myocardium (MYO)
- Aorta (AO)
- Pulmonary artery (PA)


### Importance

Cardiovascular diseases (CVDs) are the leading global cause of death, making accurate cardiac segmentation critical for:

✔ Early diagnosis of coronary artery disease, valve defects, and other conditions

✔ Precise surgical planning and intervention

✔ Improved patient outcomes through detailed anatomical modeling

While lifestyle factors (diet, smoking, exercise) influence heart health, advanced segmentation enables early detection of structural abnormalities, often before symptoms appear.

### Visualization

![image](https://github.com/user-attachments/assets/af676731-01e4-4d75-9b56-5d1dae056d7e)

*RV – right ventricle; LV – left ventricle; MYO – myocardium; RA – right atrium; LA – left atrium; AO – aorta; PA – pulmonary artery.*

![alt text](https://zmiclab.github.io/zxh/0/mmwhs/res/WholeHeartSegment_ErrorMap_WhiteBg.gif)

*Example segmentation output with error mapping*

## Installation

### Qiuck Start

Install the `hsa` (Heart Segmentation App) via pip:

```
pip install .
```

### Command Help

View all options:

```
hsa --help
```

## Usage Examples

#### Training a Model

Train a UNETR model with custom parameters:

```bash
hsa train \
  --model unetr \
  --image_dir "./image_folder" \
  --label_dir "./data/label_folder" \
  --output_dir "./runs" \
  --tag "your-run-tag" \
  --epochs 250 \
  --seed 0 \
  --dataset_config "./data/dataset.json"
```

#### Evaluation

Evaluate a trained model:

```
hsa eval \
  --model unetr \
  --checkpoint "./runs/checkpoints/checkpoint.pth" \
  --image_dir "./image_folder" \
  --label_dir "./data/label_folder" \
  --output_dir "./runs" \
  --tag "your-run-tag" \
  --dataset_config "./data/dataset.json"
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

Original Paper: https://doi.org/10.1007/978-3-030-32245-8_53

Download Link: https://www.kaggle.com/datasets/xiaoweixumedicalai/chd68-segmentation-dataset-miccai19
