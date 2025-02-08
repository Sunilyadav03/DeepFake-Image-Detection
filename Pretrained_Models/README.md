# Deepfake Image Detection Project using pre-trained models 🕵️‍♂️🔍

Welcome to the **Deepfake Image Detection Project**! This repository is dedicated to detecting deepfake images using state-of-the-art deep learning models. With the rise of deepfake technology, it has become increasingly important to develop robust methods to distinguish between real and fake images. This project fine-tunes popular convolutional neural networks (CNNs) like VGG16, VGG19, InceptionV3, and ResNet50 to achieve high accuracy in detecting deepfake images. 🚀

---

## 📌 Project Overview

This project focuses on fine-tuning pre-trained CNN models to detect deepfake images. The models are fine-tuned on a dataset of **140k real and fake faces** collected from Kaggle. The following models have been fine-tuned:

- **VGG16**: Fine-tuned on the last 5 convolutional layers.
- **VGG19**: Fine-tuned on the last 8 convolutional layers.
- **InceptionV3**: Fine-tuned on the last 8 convolutional layers.
- **ResNet50**: Fine-tuned on the last 14 convolutional layers.

The models have been trained, validated, and tested to achieve impressive results in detecting deepfake images. 🎯

---

## 🏆 Results

Here are the performance metrics of the fine-tuned models:

| Model          | Test Accuracy | Test Loss       |
|----------------|---------------|-----------------|
| **VGG16**      | 0.9527        | 0.1231          |
| **VGG19**      | 0.9521        | 0.1297          |
| **InceptionV3**| 0.7720        | 0.4770          |

These results demonstrate the effectiveness of fine-tuning pre-trained models for deepfake detection. 🎉

---

## 🛠️ Repository Structure

The repository is organized as follows:
```
deepfake-detection/Pretrained_Models
├── VGG16_finetuning.py          # Code for fine-tuning VGG16
├── VGG19_finetuning.py          # Code for fine-tuning VGG19
├── data_augmentation.py         # Code for data augmentation
├── data_preprocessing.py        # Code for data preprocessing
├── inceptionV3_finetuning.py    # Code for fine-tuning InceptionV3
├── load_pre_trained_models.py
├── plot_loss_accuracy_graph.py
├── requirements.txt             # List of dependencies
├── resnet50_finetuning.py       # Code for fine-tuning ResNet50
├── README.md                    # This file
└── datasets/                    # Directory for datasets (not included in repo)

```


---

## 🚀 Getting Started

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

You can install all the required dependencies by running:

```bash
pip install -r requirements.txt
```

#### Running the Code
**Data Preprocessing**: Run data_preprocessing.py to preprocess the dataset.

**Data Augmentation**: Run data_augmentation.py to augment the dataset.

**Model Fine-Tuning**: Run the respective fine-tuning scripts (e.g., VGG16_finetuning.py) to train the models.

**Evaluation**: The scripts will automatically evaluate the models on the test set and display the results.

---

## 📊 Dataset
The dataset used in this project consists of **140k real and fake faces** collected from Kaggle. The dataset is balanced, with an equal number of real and fake images. The fake images are generated using deepfake technology, making the dataset ideal for training and evaluating deepfake detection models.

---

## 🔍 Key Features
**Fine-Tuning**: Pre-trained models are fine-tuned on specific convolutional layers to adapt them for deepfake detection.

**Data Augmentation**: Techniques like rotation, flipping, and scaling are used to increase the diversity of the training data.

**High Accuracy**: The fine-tuned models achieve high accuracy in detecting deepfake images.

**Modular Code**: The code is modular and easy to extend for further experimentation.

---

## 📚 Research Papers and Articles
Here are some research papers and articles that inspired this project:

1. **DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection**
[Link to Paper](https://arxiv.org/abs/2001.00179)

2. **DeepFake Detection by Analyzing Convolutional Traces**
[Link to Paper](https://arxiv.org/abs/2004.10448)

3. **A Survey on Deep Learning for Detecting Deepfakes**
[Link to Paper](https://ieeexplore.ieee.org/document/9357003)

4. **The DeepFake Detection Challenge (DFDC) Dataset and Benchmark**
[Link to Article](https://arxiv.org/abs/2006.07397)

---

## 🤝 Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Let's work together to make this project even better! 🌟

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments
Kaggle for providing the dataset.

TensorFlow and Keras for providing the pre-trained models and tools for fine-tuning.

The open-source community for their invaluable contributions to deep learning research.

---

## 🌟 Show Your Support
If you find this project useful, please consider giving it a ⭐️ on GitHub. Your support motivates me to keep improving this project! 🚀

---

Happy coding! 🎉
## Let's fight deepfakes together! 💪🔍

