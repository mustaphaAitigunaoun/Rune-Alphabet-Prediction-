# Rune Alphabet Prediction Project


This project focuses on building a machine learning model to predict and classify ancient rune alphabet symbols from images. Using a dataset of rune images, the model is trained to recognize and classify different rune letters, such as Fehu, Uruz, Ansuz, and more.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)


---

## Introduction

Runes are ancient alphabets used by Germanic tribes, and each symbol carries unique meanings and historical significance. This project leverages modern machine learning techniques to classify rune symbols from images. The model is built using PyTorch and is trained on a dataset of rune images.

### Key Features
- **Image Classification**: Predicts the type of rune symbol from an input image.
- **Transfer Learning**: Uses a pretrained ResNet18 model for efficient training.
- **Data Augmentation**: Improves model generalization with techniques like rotation, flipping, and normalization.
- **Evaluation**: Provides accuracy metrics on the test set and visualizes predictions.

---

## Dataset

The dataset used for this project is titled **"Enigmatic Runes: Gateway to Ancient Wisdom"** and is available on Kaggle. It contains a collection of rune images, each labeled with its corresponding rune letter.

- **Dataset Link**: [Enigmatic Runes: Gateway to Ancient Wisdom](https://www.kaggle.com/datasets/johnrem/enigmatic-runes-gateway-to-ancient-wisdom)


---

## Project Structure

The project is organized as follows:

80% for Training ,
10% for Validation ,
10% for Testing ,

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/rune-alphabet-prediction.git
   cd rune-alphabet-prediction
   
3. **Install Dependencies**:

Ensure you have Python 3.8 or higher installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
