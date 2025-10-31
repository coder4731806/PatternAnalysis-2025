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

```diff
PatternAnalysis-2025/
│
├── LICENSE
├── README.md                        # Project documentation (this file)
│
├── dataLoader.py                    # Generates image pairs with undersampling and saves .pkl
├── dataLoader1.py                   # Colab notebook version for experimentation
├── dataLoader1.ipynb                # Colab notebook for interactive loading and debugging
│
├── siamese_network.py               # Siamese CNN architecture, model definition, training logic
├── siamese_training_plot.png        # Accuracy/Loss visualization generated after training
├── trainingScript.py                # Main training pipeline – loads data, trains model, saves weights
│
├── siamese_weights/                 # Saved model weights (.h5 files)
│   └── siamese_weights_seed42.weights.h5
│
├── subset_train/                    # Subset of ISIC 2020 images used for training
│   ├── ISIC_XXXXXX.jpg
│   └── ...
│
├── test/                            # Directory for test data (optional or future use)
│
├── Groundtruth.csv                  # Metadata linking image names to class labels (benign/malignant)
│
├── siamese_isic.pkl                 # Cached full dataset of pairs before splitting
├── siamese_isic_split.pkl           # Final balanced dataset split into train/test
├── train_pairs.pkl                  # Alternative cached version of generated training pairs
│
└── (Optional future files)
    ├── utils.py                     # Helper functions (if added later)
    ├── plots/                       # Additional experiment plots or results

```


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

### Advantages Disadvandates Of Model


### Data Augmentation 

Resizing all images to 105×105 pixels

Pixel normalization (scaling to the 0–1 range)

(Planned) Data augmentation such as rotation, flipping, and zooming

Note: Although grayscale conversion was considered for comparison, the final pipeline uses RGB color 
images (3 channels) for training.Melanoma and other skin cancers are often distinguished by color variation — subtle differences in pigmentation, redness, or bluish hues.

RGB preserves this information. Grayscale throws away hue and saturation, which can hide these features.

Example: Two lesions might have the same texture but different brown-to-black gradients — crucial for diagnosis.
Feature richness

Color images allow the network to learn more complex combinations of features (e.g., vascularity, contrast patterns, melanin depth).

### Training 

Loss: Binary Crossentropy

Optimizer: Adam (LR=0.00005)

Batch Size: 32

Epochs: 10

Early Stopping: Enabled (patience=5, min_delta=0.01)

Pretrained Weights: Optional continuation from last saved .h5

Training command:
```python 
!python siamese_train.py
```


### Training Result
images generated..

training loss curve 
increase image resolution size training loss jumps

### Model Benchmarking
mandotary ones 
BenchMark - intersection over union as an evaluation metric

### Run Instructions 

Must have the images downloaded or the pki files
```python
from google.colab import drive
drive.mount('/content/drive')

loader = DataLoader(width=105, height=105, channels=3,
                    data_path="/content/drive/MyDrive/subset_train",
                    csv_path="/content/drive/MyDrive/Groundtruth.csv",
                    output_path="/content/drive/MyDrive/siamese_isic_split.pkl")
loader.load()


!python siamese_train.py

```

### Dependencies 

Python 3.12+
TensorFlow 2.17+
NumPy
Pandas
Pillow
Matplotlib
scikit-learn

