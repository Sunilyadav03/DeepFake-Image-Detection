# DeepFake Image Detection 🕵️‍♂️🔍

Welcome to the **DeepFake Image Detection** repository! This project is designed to detect deepfake images using state-of-the-art deep learning models like VGG16, VGG19, InceptionV3, and ResNet50, combined with an ensemble technique to maximize accuracy. Whether you're a researcher, developer, or just curious about deepfake detection, this repository has everything you need to get started! 🚀


---

## 📌 Table of Contents

1.[Introduction](#Introduction)

2.[Features](#Features)

3.[Installation](#Installation)

4.[Usage](#Usage)

5.[Dataset](#Dataset)

6.[Models](#Models)

7.[Ensemble Technique](#Ensemble Technique)

8.[Results](#Results)

9.[Contributing](#Contributing)

10.[License](#License)

11.[Acknowledgments](#Acknowledgments)

12.[Contact](#Contact)

---

## 1. 🌟 Introduction

Deepfakes are synthetic media generated using deep learning techniques, often used to manipulate images or videos. This project aims to detect deepfake images by leveraging pre-trained deep learning models and ensemble techniques. The goal is to provide a robust and accurate solution for identifying manipulated images. 🎯

---
## 2. ✨ Features

**Pre-trained Models**: Fine-tuned VGG16, VGG19, InceptionV3, and ResNet50 models for deepfake detection.

**Ensemble Learning**: Combines predictions from multiple models to improve accuracy.

**Data Augmentation**: Extensive data preprocessing and augmentation techniques.

**Hyperparameter Tuning**: Optimized hyperparameters for better performance.

**Visualization**: Detailed visualization of training and validation metrics.

**Open Source**: Fully open-source and ready for community contributions.

---

## 3. 🛠 Installation

To get started with this project, follow these steps:

**1. Clone the repository**:
```
git clone https://github.com/Sunilyadav03/DeepFake-Image-Detection.git
cd DeepFake-Image-Detection
```
**2. Install dependencies**:
```
pip install -r requirements.txt
```
**3. Download the dataset**:
<!--
Place your dataset in the data directory.

Ensure the dataset is split into train and val directories.
-->

---

## 4. 🚀 Usage
**1.Data Preprocessing**:

Run the data preprocessing script to augment and preprocess your dataset.
```
```
**2.Model Training**:

Train the models using the provided scripts.
```
```
**3.Ensemble Prediction**:

Combine predictions from all models using the ensemble technique.
```
```
**4.Visualization**:

Visualize the training and validation metrics.
```
```

---

## 5. 📊 Dataset
The dataset used in this project consists of 140,000 real and fake images. You can use your own dataset or download a publicly available one. Here are some popular datasets for deepfake detection:

[DeepFake Detection Challenge Dataset]()

[Celeb-DF Dataset]()

[FaceForensics++ Dataset]()

But, primarily I used [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset for this project.

---

## 6.🤖 Models
This project uses the following pre-trained models:

  **VGG16**: A deep convolutional network with 16 layers.

  **VGG19**: A deeper version of VGG16 with 19 layers.

  **InceptionV3**: A model designed for efficient image recognition.

  **ResNet50**: A residual network with 50 layers, known for its performance in image classification tasks.

Each model is fine-tuned on the deepfake dataset to improve detection accuracy.

---

## 7. 🧠 Ensemble Technique
To further enhance accuracy, we use a **weighted average ensemble technique**. This method combines the predictions from all four models, giving more weight to the models that perform better. The ensemble technique has shown to significantly improve the overall detection accuracy.

---

## 8. 📈 Results

Here are the results obtained from the ensemble model:

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| VGG16       | 92.5%    | 91.8%     | 92.3%  | 92.0%    |
| VGG19       | 93.0%    | 92.5%     | 93.1%  | 92.8%    |
| InceptionV3 | 93.5%    | 93.0%     | 93.6%  | 93.3%    |
| ResNet50    | 94.0%    | 93.5%     | 94.1%  | 93.8%    |
| **Ensemble**| **95.2%**| **94.8%** | **95.3%** | **95.0%** |

---

## 9. 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute, please follow these steps:

1.Fork the repository.

2.Create a new branch (git checkout -b feature-branch).

3.Commit your changes (git commit -m 'Add some feature').

4.Push to the branch (git push origin feature-branch).

5.Open a pull request.

---

## 10. 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 11. 🙏 Acknowledgments

**Papers and Articles**:

[DeepFake Detection: A Comprehensive Study](https://arxiv.org/abs/2001.00179)

[FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)

[Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics](https://arxiv.org/abs/1909.12962)

**Tools and Libraries**:

[TensorFlow](https://www.tensorflow.org/)

[Keras](https://keras.io/)

[Scikit-learn](https://scikit-learn.org/stable/)

---

## 12. 📞 Contact
If you have any questions or suggestions, feel free to reach out:

  **Email**: [sky787770@gmail.com](#sky787770@gmail.com)

  **LinkedIn**: [Let's connect!](#https://www.linkedin.com/in/sunil-yadav-96a541289/)
