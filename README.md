# Project Title

Official resource for the paper:  
**"FAN-Enhanced Vision Transformer and Distributed Bidirectional Optical Flow for Heart Failure Detection from Echocardiography Videos"**

## Overview

This repository contains the source code and preprocessed data used in our research paper. The dataset in this repository is preprocessed using Algorithm 1 in the paper to remove text overlay from the video. The raw video contains confidential information about the subject, such as the subject's name, medical record, etc. The raw dataset is available upon reasonable request to the corresponding author (Mgs. M. Luthfi Ramadhan: luthfiramadhan@unsri.ac.id) and is subject to evaluation by Dr. Lies Dina Liastuti.

---

## Project Structure

```text
.
├── data/                # Dataset directory (Already preprocessed using Algorithm 1 to remove text overlay)
├── fan.py               # FAN layer
├── main.py              # Main training and evaluation script
├── main.sh              # Shell script to run grid search
├── preprocessing.py     # Data preprocessing
├── transformer.py       # Transformer model
├── vit.py               # Vision Transformer (ViT) model
└── README.md
