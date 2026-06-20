# Handwritten Digit Recognition using CNN

A simple handwritten digit recognition system built using **PyTorch** and deployed with **Streamlit**. The model is trained on the **MNIST dataset** and predicts digits from 0 to 9 from uploaded images.

---

## Features

- Train a CNN model on the MNIST dataset
- Achieve around 98–99% test accuracy
- Save the trained model for inference
- Upload handwritten digit images through a Streamlit web app
- Predict digits from 0–9 in real time

---

## Project Structure

```
mnist-digit-recognition-cnn/
│
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
└── mnist_cnn.pth
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow

---

## CNN Architecture

```
Input (1 × 28 × 28)
        ↓
Conv2D (32 filters)
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D (64 filters)
        ↓
ReLU
        ↓
MaxPool
        ↓
Flatten
        ↓
Linear (128)
        ↓
Dropout
        ↓
Linear (10)
        ↓
Digit Prediction
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python train.py
```

This will:
- Download the MNIST dataset
- Train the CNN model
- Evaluate it on the test set
- Save the trained model as:

```
mnist_cnn.pth
```

---

## Running the Streamlit App

Run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How It Works

```
Image Upload
     ↓
Preprocessing
     ↓
CNN Model
     ↓
Probability Scores
     ↓
Predicted Digit
```

---
## Evaluation

- Assessed model performance using standard evaluation metrics to ensure reliable classification outcomes.
- Performed prediction analysis to identify common digit misclassifications and interpret model behavior.
## Resume Description

- Built a deep learning-based handwritten digit recognition system for automated classification of grayscale handwritten digits.
- Implemented and evaluated a Convolutional Neural Network (CNN) on the MNIST dataset for digit classification.
- Assessed model performance through prediction analysis and standard evaluation metrics for reliable classification outcomes.
- Analyzed prediction outcomes to identify common misclassification patterns and interpret model behavior effectively.

---

## Future Improvements

- Add a drawing canvas for users to draw digits directly.
- Deploy the application online.
- Visualize prediction probabilities.
- Experiment with deeper CNN architectures.

---

## Author

Developed as part of internship interview preparation and deep learning practice.