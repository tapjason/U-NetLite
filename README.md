# Medical Image Segmentation with Enhanced U-Net

This repository implements an **Enhanced U-Net** architecture with lightweight attention mechanisms for efficient 2D medical image segmentation. The project focuses on improving segmentation accuracy while minimizing computational complexity, making it ideal for deployment in resource-constrained environments.


## Project Overview

Medical image segmentation plays a critical role in healthcare applications such as diagnosis, surgical planning, and treatment monitoring. This project builds upon the standard U-Net architecture by introducing the following enhancements:

- **Squeeze-and-Excitation (SE) blocks** for channel attention  
- **Depthwise separable convolutions** for parameter efficiency  
- **Spatial attention mechanisms** to focus on salient regions  
- **Grouped convolutions** for ultra-lightweight performance  

All models are evaluated on the **ISIC 2018 Skin Lesion Segmentation** and the **Breast Ultrasound Images Dataset** datasets.


## Features

- Baseline U-Net implementation
- U-Net with **Depthwise Separable Convolutions**  
- U-Net with **Squeeze-and-Excitation (SE) blocks**  
- Combined SE + Depthwise U-Net  
- U-Net with **Spatial Attention**  
- Ultra-Lightweight U-Net using **Grouped Convolutions**  
- Evaluation using Dice, IoU, and Pixel Accuracy  
- Support for dermoscopic images (ISIC 2018)
- Support for ultrasound images (BUSI)  


## Setup and Installation

### Step 1: Create Environment

```bash
conda create -n medseg python=3.10 -y
conda activate medseg
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage Instructions
### Step 1: Data Preparation and Exploration

Run the following notebook to download, process, and store both ISIC and BUSI datasets:
```
notebooks/data_extraction.ipynb
```
The notebook assumes that `kaggle.json` credentials are provided for the Kaggle API, but also includes links to manual installation where necessary.
### Step 2: Model Training
Train your model variants on each dataset using:
```bash
# For ISIC dataset
notebooks/isic_models.ipynb

# For BUSI dataset
notebooks/busi_models.ipynb
```

During training, checkpoints are saved to `saved_models/` for both `ISIC/` and `BUSI/` datasets. The best-performing model on the validation set is saved as `best_model.pth`, and the final model after training as `final_model.pth`.
### Step 3: Model Evaluation


```bash
notebooks/model_evaluation.ipynb
```



## Project Structure
```
U-NetLite/
├── notebooks/ # Jupyter notebooks for pipeline stages  
│ ├── data_extraction.ipynb # Load, preprocess, and save ISIC + BUSI datasets  
│ ├── isic_models.ipynb # Train model variants on ISIC dataset  
│ ├── busi_models.ipynb # Train model variants on BUSI dataset  
│ └── model_evaluation.ipynb # Visualize predictions and compare metrics  
│  
├── data/ # Datasets  
│ ├── isic_2018_task1_data/ # Raw ISIC input and mask folders  
│ │ ├── ISIC2018_Task1-2_Training_Input/  
│ │ ├── ISIC2018_Task1-2_Test_Input/  
│ │ └── ISIC2018_Task1_Training_GroundTruth/  
│ └── Dataset_BUSI_with_GT/ # Raw BUSI ultrasound dataset  
│ │ ├── benign/  
│ │ ├── malignant/  
│ │ └── normal/  
│  
├── src/ # Source code modules  
│ ├── models/  
│ │ ├── unet.py # Baseline U-Net architecture  
│ │ ├── convolutions.py # Depthwise/grouped conv blocks  
│ │ ├── attention.py # SE and spatial attention modules  
│ │ └── enhanced_unet.py # Combined enhanced U-Net variants  
│ └── utils/  
│ │ ├── metrics.py # Dice, IoU, pixel accuracy  
│ │ └── losses.py # BCE+Dice loss function  
│ └── data/ # Contains dataset loading, preprocessing, and augmentations for each dataset
│ │ ├── isic_dataset.py 
│ │ └── busi_dataset.py
│  
├── saved_models/  # Auto-saved training outputs
│   ├── isic/
│ │ ├── checkpoint.pth          # Periodic training checkpoints
│ │ ├── best_model.pth          # Best model based on validation Dice
│ │ └── final_model.pth          # Last model after full training
│   └── busi/
│ │ ├── checkpoint.pth
│ │ ├── best_model.pth
│ │ └── final_model.pth
├── results/ # Prediction visualizations and metrics
```
