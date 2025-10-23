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


```diff 
    ┌───────────────────────┐                               ┌───────────────────────┐
    │      Input Image 2     │                              │      Input Image 1     │
    └───────────┬───────────┘                               └────────────┬───────────┘
                │                                                        │
                ▼                                                        ▼
               ┌───────────────────────────────────────────────────────────┐
               │               Shared Convolutional Neural Net             |
               │                (same weights for both branches)           |
               └────────────────┬──────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
        ▼                                               ▼
┌───────────────────────┐                    ┌───────────────────────┐ 
│  CNN Layers            │                   │  CNN Layers            │
│  (Conv → BatchNorm →   │                   │  (Conv → BatchNorm →   │
│   Activation → Dropout │                   │   Activation → Dropout │
│   → Pooling)           │                   │   → Pooling)           │
└─────────────┬─────────┘                    └─────────────┬─────────┘
              │                                              │
              ▼                                              ▼
      ┌──────────────────-┐                          ┌-──────────────────┐
      │   Feature Vector  │                          │   Feature Vector  │
      │       (F1)        │                          │       (F2)        │
      └────────┬──────────┘                          └────────┬──────────┘
               │                                              │
               └──────────────┬───────────────────────────────┘
                              ▼
                 ┌───────────────────────────┐
                 │   Distance Calculation     │
                 │ (Euclidean or L1 distance) │
                 └────────────┬───────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │     Contrastive Loss       │
                 │ (Pulls similar pairs close │
                 │   and pushes apart others) │
                 └────────────┬───────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │     Model Training         │
                 │  (with Early Stopping,     │
                 │   Dropout, BatchNorm)      │
                 └────────────┬───────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │        Evaluation          │
                 │ (Accuracy, Loss, Similarity│
                 │  Scores on Test Pairs)     │
                 └───────────────────────────┘
``` 


### Data Preperation
The DataLoader class constructs pairs of images for training the Siamese network.

Each pair consists of:
A positive pair: two images from the same class (benign–benign or malignant–malignant).
A negative pair: two images from different classes (benign–malignant).
Since the ISIC 2020 dataset is imbalanced (far more benign than malignant images), the pairing strategy had to be 
adjusted to prevent biased learning.

```python
with open(self.output_path, "wb") as f:
    pickle.dump([[x1, x2], y], f)
```
Steps:

1. The _make_pairs() method:

2. Groups images by class (target = 0 benign, 1 malignant).

3. Randomly selects pairs to create both positive and negative examples.

4, Applies undersampling to balance the dataset — ensuring equal counts of positive and negative pairs.

5. Normalizes and resizes each image before saving to a .pkl file for faster loading in Colab.

Each image creates a positive pair with a random image of the same class and a negative pair with the opposite class.
Then the positive and negative pairs are concantenanted together into pairs of variable corresponding to 0 or 1 labels.
```python
x_first_balanced, x_second_balanced, y_balanced = self._make_pairs()
pickle.dump([[x_first_balanced, x_second_balanced], y_balanced], f)
```
The method takes in two inputs, x and y, and groups the images by class (malignant vs. benign). For each iteration, 
it selects a positive pair and then creates a negative pair. Both images are loaded, resized, and normalized, then
stored in x_first, x_second, and y (representing the similarity score, 1 or 0). Finally, the data is converted into 
NumPy arrays.

![positive-negative pair images](positivenegativepair.png)

Train and Validation split: To create the training and validation sets, the pairs of images were split 80/20, 
with 80% used for training and 20% reserved for model validation. This ensures sufficient data is available to accurately 
train the model and set aside a validation dataset to assess performance during training and identify potential overfitting.



### Data Set

dimentionality and channels 
training and testing

### Project Goals

Build a Siamese CNN for melanoma similarity learning with an accuracy of at least 85%.

Mitigate class imbalance through data-level balancing.

Evaluate model performance on unseen pairs.

Visualize feature learning and training metrics.

### File Structure




### Model Architecture

The architecture follows the original Siamese CNN (Koch et al., 2015):

Four convolutional blocks with Batch Normalization and MaxPooling.

Flatten → Dense(4096) feature embedding layer.

Distance computation via absolute difference (L1).

Output: binary similarity score using a Sigmoid activation.

Additional techniques:

L2 regularization to prevent overfitting.

Dropout layers to increase robustness.

Adam optimizer with low learning rate (5e-5).
