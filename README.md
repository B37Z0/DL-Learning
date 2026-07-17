# DL-Learning

A collection of deep learning projects.

## MLPs + CNNs

| Project | Description | Model | Dataset |
|---|---|---|---|
| [XOR Simple MLP](MLPs%20+%20CNNs/NumPy%20-%20XOR%20Simple%20MLP) | Binary classification on XOR - MLP from scratch | 2→4→1 MLP (NumPy) | XOR truth table |
| [LeNet-5](MLPs%20+%20CNNs/PyTorch%20-%20LeNet5) | Handwritten digit classification - classic LeNet-5 architecture | LeNet-5 (Conv→Pool→Conv→Pool→FC) | MNIST |
| [CIFAR-10 CNN](MLPs%20+%20CNNs/PyTorch%20-%20CIFAR10%20CNN) | 10-class color image classification | 2-conv + 3-FC network | CIFAR-10 |
| [FashionMNIST MLP](MLPs%20+%20CNNs/PyTorch%20-%20FashionMNIST%20MLP) | 10-class clothing image classification | 784→512→512→10 MLP | FashionMNIST |
| [Titanic MLP](MLPs%20+%20CNNs/PyTorch%20-%20Titanic) | Kaggle Titanic survival prediction | Entity embedding + 3-layer MLP | Kaggle Titanic |
| [Transfer Learning — ResNet-18](MLPs%20+%20CNNs/PyTorch%20-%20CV%20Transfer%20Learning/EfficientNet-B0) | Binary classification using transfer learning (fine-tuning vs. fixed feature extraction) | ResNet-18 (pretrained ImageNet) | Hymenoptera (ants vs bees) |
| [Transfer Learning — EfficientNet-B0](MLPs%20+%20CNNs/PyTorch%20-%20CV%20Transfer%20Learning/ResNet-18) | Binary classification using transfer learning (fine-tuning vs. fixed feature extraction) | EfficientNet-B0 (pretrained ImageNet) | CIFAR-10 |

## RNNs

| Project | Description | Model | Dataset |
|---|---|---|---|
| [MovingMNIST LSTM](RNNs/PyTorch%20-%20MovingMNIST%20LSTM) | Video prediction - observe 10 frames, predict next 10 autoregressively | Frozen ResNet-18 encoder → LSTM → ConvTranspose decoder | MovingMNIST |

## CV

| Project | Description | Model | Dataset |
|---|---|---|---|
| [YOLO Real-Time OD](CV/YOLO%20-%20Real%20Time%20OD) | Real-time object detection from webcam | YOLOv26s | COCO (pretrained) |
