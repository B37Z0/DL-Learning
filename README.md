# DL-Learning

A collection of deep learning projects.

## MLPs + CNNs

| Project | Description | Model | Dataset |
|---|---|---|---|
| [XOR Simple MLP](MLPs+++CNNs/NumPy+-+XOR+Simple+MLP) | Binary classification on XOR - MLP from scratch | 2→4→1 MLP (NumPy) | XOR truth table |
| [FashionMNIST MLP] | 10-class clothing image classification | 784→512→512→10 MLP | FashionMNIST |
| [LeNet-5] | Handwritten digit classification - classic LeNet-5 architecture | LeNet-5 (Conv→Pool→Conv→Pool→FC) | MNIST |
| [CIFAR-10 CNN] | 10-class color image classification | 2-conv + 3-FC network | CIFAR-10 |
| [Titanic MLP] | Kaggle Titanic survival prediction | Entity embedding + 3-layer MLP | Kaggle Titanic |
| [Transfer Learning — ResNet-18] | Binary classification using transfer learning (fine-tuning vs. fixed feature extraction) | ResNet-18 (pretrained ImageNet) | Hymenoptera (ants vs bees) |
| [Transfer Learning — EfficientNet-B0] | Binary classification using transfer learning (fine-tuning vs. fixed feature extraction) | EfficientNet-B0 (pretrained ImageNet) | CIFAR-10 |

## RNNs

| Project | Description | Model | Dataset |
|---|---|---|---|
| [MovingMNIST LSTM] | Video prediction - observe 10 frames, predict next 10 autoregressively | Frozen ResNet-18 encoder → LSTM → ConvTranspose decoder | MovingMNIST |

## CV

| Project | Description | Model | Dataset |
|---|---|---|---|
| [YOLO Real-Time OD] | Real-time object detection from webcam | YOLOv26s (Ultralytics) | COCO (pretrained) |
