# LARGO: Flower Image Generation using DDPM

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for generating flower images from the Oxford Flowers dataset.


##  Overview

This project implements a conditional diffusion model (DDPM) for generating flower images. It includes:
-  UNet architecture with attention mechanisms
- Support for class-conditional generation
- Training and inference pipelines
- Web interface using Streamlit
- Google Colab integration



##  Features

- **Conditional Generation**: Generate specific flower classes (1-102)
- **Web Interface**: User-friendly Streamlit app for real-time generation
- **Checkpoint Compatibility**: Utilities for loading models from different training setups
- **Distributed Training**: Support for multi-GPU training with DDP


##  Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for training 

### Step 1: Clone Repository
```bash
git clone https://github.com/Yassin-Zaher/GenAI-MiniProjet.git
cd GenAI-MiniProjet
```


### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
Download the Oxford Flowers dataset :
```bash
mkdir -p dataset
# Download and extract Oxford 102 Flowers dataset

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
```
## Project structure
```text
largo-diffusion/
├── dataset.py           # Dataset management and preprocessing
├── unets.py             # UNet model architecture
├── main.py              # Main training and inference engine
├── app.py               # Streamlit web interface
├── colab.ipynb          # Google Colab code
├── checkpoints/         # Model weights (.pt files)
├── sampled_images/      # Generated images
└── requirements.txt     # Python dependencies
```
## Usage
### Training
Basic Training Command
```bash
python main.py \
    --arch UNet \
    --dataset flowers \
    --class-cond \
    --diffusion-steps 1000 \
    --sampling-steps 250 \
    --batch-size 128 \
    --lr 0.0001 \
    --epochs 500 \
    --ema_w 0.9995 \
    --data-dir ./dataset/ \
    --save-dir ./trained_models/ \
    --seed 112233
```

### Inference/Sampling
Generate Random Flowers
```bash
python main.py \
    --arch UNet \
    --dataset flowers \
    --class-cond \
    --sampling-only \
    --sampling-steps 1000 \
    --num-sampled-images 10 \
    --ddim \
    --batch-size 8 \
    --save-dir ./sampled_images/ \
    --pretrained-ckpt ./checkpoints/ema_model.pt \
    --num-sampled-images 16 \
    --sampling-steps 250 \
    --save-dir ./sampled_images
  ```
Generate Specific Flower Class
```bash
python main.py \
    --arch UNet \
    --dataset flowers \
    --class-cond \
    --specific-class 51 \
    --sampling-only \
    --sampling-steps 1000 \
    --num-sampled-images 10 \
    --ddim \
    --batch-size 8 \
    --save-dir ./sampled_images/ \
    --pretrained-ckpt ./checkpoints/ema_model.pt \
    --num-sampled-images 16 \
    --sampling-steps 250 \
    --save-dir ./sampled_images
  ```

## Web Interface
Launch Streamlit App
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

**Web Interface Features:**
Adjust number of images (1-16)

Control sampling steps (50-1000)

Toggle class conditioning

Select specific flower class (1-102)

Real-time generation and display

##  Google Colab
### Quick Start
1. Open colab.ipynb in Google Colab

2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Run all cells sequentially

### Generate Samples in Colab
```python
# In a Colab cell:
!python main.py \
  # rest of the arguments
  ```
Deploy Web App from Colab
```python
# Install ngrok and run Streamlit
!pip install pyngrok
!streamlit run app.py --server.port=8501 --server.address=0.0.0.0 > /dev/null 2>&1 &

# Create public URL
from pyngrok import ngrok, conf

NGROK_AUTH_TOKEN = TOKEN

conf.get_default().auth_token = NGROK_AUTH_TOKEN

public_url = ngrok.connect(8501)
print(f"Streamlit app running at: {public_url}")
```
##  Architecture

## Core Components
- **Data Layer (`data.py`)**: Handles Oxford Flowers dataset loading and preprocessing  
- **Model Layer (`unets.py`)**: Conditional U-Net with attention mechanisms  
- **Engine Layer (`main.py`)**: Diffusion process and training orchestration  
- **Interface Layer (`app.py`)**: Streamlit web interface  

## Diffusion Process
- **Forward Process**: Adds noise to images using cosine schedule  
- **Reverse Process**: Iterative denoising over 50–1000 steps  
- **Conditioning**: Timestep embeddings + class labels  

## Model Specifications
- **Base Architecture**: U-Net with skip connections  
- **Attention**: Multi-head self-attention at multiple resolutions  
- **Normalization**: GroupNorm with scale-shift conditioning  
- **Activation**: SiLU (Swish)  
- **Embeddings**: Sinusoidal timestep + learned class embeddings  

---

##  Configuration

### Key Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--arch` | Neural network architecture | Required |
| `--class-cond` | Train class-conditioned diffusion model | `False` |
| `--diffusion-steps` | Number of timesteps in diffusion process | `1000` |
| `--sampling-steps` | Number of timesteps in diffusion process | `250` |
| `--ddim` | Sampling using DDIM update step | `False` |
| `--dataset` | Dataset name | Required |
| `--data-dir` | Dataset directory | `./dataset/` |
| `--batch-size` | Batch size per GPU | `128` |
| `--lr` | Learning rate | `0.0001` |
| `--epochs` | Number of training epochs | `500` |
| `--ema_w` | EMA weight for model averaging | `0.9995` |
| `--pretrained-ckpt` | Pretrained model checkpoint path | `None` |
| `--delete-keys` | Keys to delete from pretrained checkpoint | `None` |
| `--sampling-only` | No training, just sample images | `False` |
| `--num-sampled-images` | Number of images to sample | `50000` |
| `--save-dir` | Directory to save models/images | `./trained_models/` |
| `--local_rank` | Local rank for distributed training | `0` |
| `--seed` | Random seed | `112233` |
| `--specific-class` | Generate images of only this class label | `None` |
| `--p_drop` | Probability of dropping labels for classifier-free guidance | `0.0` |



---

##  Results

Generated flower images exhibit:
- Realistic petal structures and colors  
- Class-consistent features  
- High diversity within classes  
- Resolution up to **128×128** pixels  

### Sample Generations
```text
./sampled_images/
```

#### View the images
```python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

sample_dir = './sampled_images'
npz_files = sorted(glob.glob(os.path.join(sample_dir, '*.npz')))

if not npz_files:
    print("No .npz files found in ./sampled_images/")
else:
    for npz_file in npz_files:
        print(f"\n--- Loading {os.path.basename(npz_file)} ---")
        data = np.load(npz_file)
        images = data['arr_0']  # Shape: (N, H, W, C)
        labels = data['arr_1']  # Shape: (N,)

        num_images = len(images)
        cols = min(5, num_images)
        rows = int(np.ceil(num_images / cols))

        plt.figure(figsize=(cols * 2.5, rows * 2.5))
        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            # Ensure image is in [0, 255] and uint8
            img = images[i]
            if img.dtype != np.uint8:
                img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.title(f"Class: {labels[i]}")
            plt.axis('off')
        plt.suptitle(f"From: {os.path.basename(npz_file)}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
```
##  License

This project is for academic purposes as part of a Master's program.  
Feel free to use this project.


##  Authors

For questions or collaborations:

- **Bahlaouane Salaheddine**: [GitHub](https://github.com/Skeiba)
- **Zaher Yassin**: [GitHub](https://github.com/Yassin-Zaher)

Master in Computer Science and Artificial Intelligence • 2025–2026