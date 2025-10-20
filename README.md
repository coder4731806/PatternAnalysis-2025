# Siamese Networks
### Skin Lesion Analysis: Melanoma Detection  using The ISIC 2020 Challenge Dataset

--------------------------------------------------------------------------------------

Author: Harshit Vishwa s47318063

### Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Dataset Details](#dataset-details)
- [Project Goals](#project-goals)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training Process](#training-process)
- [Training Results](#training-results)
- [Model Benchmarking](#model-benchmarking)
- [Run Instructions](#run-instructions)
- [Dependencies](#dependencies)


### Introduction To Siamese Neural Network
A Siamese neural network is a neural network architecture that learns to measure the similarity between two inputs 
instead of classifying them. It uses two identical subnetworks with shared weights to compare pairs of inputs and output
how alike they are. In my project, they are used to detect cancerous and non-cancerous skin lesions. The outputs are 
compared using a contrastive loss function, which pulls similar pairs closer together and pushes dissimilar pairs 
farther apart. This allows the network to learn a more meaningful feature representation of the input data.
