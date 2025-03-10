# Traffic Sign Classification using Convolutional Neural Networks

## Project Overview

This project aims to build a Convolutional Neural Network (CNN) for classifying road signs based on images. The goal is to create a model that can accurately identify different types of road signs, an essential task in autonomous vehicles and intelligent transportation systems.

## Objective

The main objective of this project is to develop a deep learning model that:
- Takes images of road signs as input.
- Classifies them into appropriate categories.
- Achieves a high level of accuracy in recognizing various road signs.

## Approach

The approach taken in this project involves building and training a CNN using a road sign dataset. The CNN architecture includes several convolutional layers for feature extraction followed by dense layers for classification. The model's performance is evaluated based on its accuracy on a validation set.

## Technologies Used

- Python 3.12
- TensorFlow/Keras
- Scikit-learn

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yzaazaa/trafficSignRecognition
   cd trafficSignRecognition
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:
   ```bash
   python traffic.py [ dataset ] [ modelname --optional ]
   ```

## Dataset

The dataset used in this project contains images of various road signs. Each image is labeled with the correct road sign type (e.g., "Stop," "Speed Limit," etc.). You can download the dataset from [Dataset Source](#https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip).

# Project Insights & Observations

## Key Observations

### 1. **More Layers and Parameters Lead to Higher Accuracy**
   - **Observation**: I noticed that increasing the number of layers and parameters in the model often led to a significant improvement in accuracy. Adding more layers allows the network to capture more complex features, which is especially important for tasks that require learning from intricate patterns in the data.
   - **Explanation**: Deeper models have more capacity to learn from the dataset, leading to a better fit. However, this needs to be balanced with the risk of overfitting, especially with smaller datasets.

### 2. **Balancing the Number of Weights in Flattened and Dense Layers**
   - **Observation**: I observed that it's important to try to approach the number of weights in the last flattened layer to match the first dense layer. This helps ensure that the model doesn't get bottlenecked in the transition from the convolutional layers to the fully connected layers.
   - **Explanation**: If the number of weights in the first dense layer is too high compared to the flattened layer, it can result in inefficient training. Ensuring that the number of parameters in these layers is balanced helps in improving the overall performance.

### 3. **Dropout and Small Datasets**
   - **Observation**: I found that using large dropout rates is not ideal for small datasets. With small datasets, a large dropout rate can cause the model to lose too much information during training, which harms its ability to generalize well.
   - **Explanation**: Dropout is a regularization technique used to prevent overfitting. However, with limited data, excessive dropout can prevent the model from learning useful patterns effectively. A moderate dropout rate is recommended for smaller datasets.

## Summary of Results

In my experiments, I tuned the model by adjusting the number of layers, neurons, and dropout rates. Here are the key points I learned:

- **Larger models** (with more layers and parameters) often resulted in better performance, but this comes at the cost of computational resources and risk of overfitting if the dataset is too small.
- **Balancing the number of weights** between the last flattened layer and the first dense layer ensures efficient learning and prevents bottlenecks.
- **Smaller dropout rates** worked better for smaller datasets, as large dropout rates in such cases can hinder model performance by discarding too much data.

## Tips for Future Work

- When dealing with a **small dataset**, try starting with a smaller model architecture and experiment with the dropout rate.
- Focus on tuning the **number of layers and neurons**, but be mindful of **overfitting**. Use techniques like early stopping or cross-validation to prevent it.