# 👗 Fashion-MNIST Classification with Uncertainty (Laplace Approximation) 🎯

## 📝 Description
CNN classifier for Fashion-MNIST with Laplace Approximation to quantify predictive uncertainty. Provides both class predictions and confidence estimates.

🧠 Theory

Laplace Approximation: A Bayesian technique that approximates the posterior distribution of model weights using a Gaussian around the Maximum A Posteriori (MAP) estimate. This allows the model to not only predict classes but also estimate uncertainty for each prediction, which is useful for risk-sensitive applications ⚠️.

📦 Dependencies
torch
torchvision
numpy
matplotlib
laplace-torch

## 💡 Motivation
Most deep learning models make predictions without indicating how confident they are. 
This project demonstrates **uncertainty quantification** using Laplace Approximation on a CNN classifier, which is important for applications like healthcare 🏥, autonomous systems 🚗, and other risk-sensitive domains.

## 📊 Dataset
- **Dataset:** Fashion-MNIST
- **Training images:** 60,000
- **Test images:** 10,000
- **Image size:** 28x28 grayscale
- **Classes:** T-shirt/top 👕, Trouser 👖, Pullover 👚, Dress 👗, Coat 🧥, Sandal 👡, Shirt 👔, Sneaker 👟, Bag 👜, Ankle boot 👢

## ⚙️ Features
- CNN model with 2 convolutional layers + 2 fully connected layers
- Training loop with Adam optimizer and Cross-Entropy Loss
- Evaluation on test dataset with accuracy metrics ✅
- Visualizes predictions with confidence scores 📈
- Applies Laplace Approximation to estimate predictive uncertainty 🔍
- Plots uncertainty (entropy) for individual predictions
