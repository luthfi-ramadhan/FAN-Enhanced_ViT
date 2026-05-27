# FAN-Enhanced Vision Transformer and Distributed Bidirectional Optical Flow for Heart Failure Detection from Echocardiography Videos

Official resource for the paper:  
**"FAN-Enhanced Vision Transformer and Distributed Bidirectional Optical Flow for Heart Failure Detection from Echocardiography Videos"**

## Overview

This repository contains the source code and processed data used in our research paper. The dataset in this repository is already processed using Algorithm 1 to remove text overlay from the video. The raw video contains confidential information about the subject, such as the subject's name, medical record, etc. The raw dataset is available upon reasonable request to the corresponding author (Mgs. M. Luthfi Ramadhan: luthfiramadhan@unsri.ac.id) and is subject to evaluation by Dr. Lies Dina Liastuti.

---

## Project Structure

```text
.
├── data/                # Dataset directory (Already processed using Algorithm 1 to remove text overlay)
├── fan.py               # FAN layer
├── main.py              # Main training and evaluation script
├── main.sh              # Shell script to run grid search. This script repeatedly runs main.py with various hyperparameters
├── preprocessing.py     # Data preprocessing pipeline. This includes Distance-Based Frame Selection (Algorithm 2) and Distributed Bidirectional Optical Flow (Algorithm 3)
├── transformer.py       # Transformer model
├── vit.py               # Vision Transformer (ViT) model
├── 4ch_result.xlsx      # Excel file to log every experiment result in A4C view
├── 2ch_result.xlsx      # Excel file to log every experiment result in A2C view
├── plax_result.xlsx     # Excel file to log every experiment result in plax view
└── frame_refocusing.py  # Frame refocusing (Algorithm 1). If you are planning to use the processed dataset in this GitHub, you no longer need to run this code.
```

## Software Environment
The preprocessing pipeline is mainly implemented using **OpenCV 4.11.0**. For deep learning experiments, we utilized the official NVIDIA TensorFlow deep learning container tagged with:

```text
23.09-tf2-py3
