# K-Nearest Neighbors (K-NN) Classification Project

## Overview

This project implements a **K-Nearest Neighbors (K-NN)** algorithm from scratch and provides a user-friendly **graphical user interface (GUI)** built with `Tkinter`. The project allows users to load a dataset, train a K-NN model, visualize data distribution, and evaluate the model using a confusion matrix and classification report.

### Features

- **Custom K-NN Algorithm**: Implements the K-NN algorithm for classification tasks.
- **Data Visualization**: Automatically visualizes data using scatter plots or histograms based on the number of features.
- **Confusion Matrix**: Displays the confusion matrix of the trained model.
- **Accuracy & Report**: Shows accuracy and detailed classification report (precision, recall, F1-score).
- **User-Friendly GUI**: Load datasets, visualize, train models, and see results all in one interface.

---

## Requirements

To run this project, you need to install the required Python libraries. You can install the dependencies using the provided `requirements.txt` file.

**Required Libraries:**

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tkinter` (comes pre-installed with Python)

---

## Installation

### 1. Clone the Repository

First, clone the repository from GitHub to your local machine:

```bash
git clone https://github.com/your-username/knn_project.git
cd knn_project
```

## How to Use

### 1. Load Dataset
- Open the application and click the **Load Dataset** button.
- Select a CSV file (e.g., `dataset.csv`).

### 2. Train and Test the Model
- After loading the dataset, click the **Train and Test Model** button.
- The model will automatically train on the dataset and display the following:
  - **Data Distribution**: A scatter plot or histogram based on the dataset.
  - **Confusion Matrix**: Visualizes the classification results.
  - **Accuracy**: Shows the accuracy of the model.
  - **Classification Report**: Provides precision, recall, F1-score for each class.

---

## Dataset

The project includes a synthetic dataset (`dataset.csv`) generated using `sklearn.datasets.make_classification`. It has the following structure:

- **Features**: 4 features (`feature1`, `feature2`, `feature3`, `feature4`)
- **Target**: Binary target variable (`0` or `1`)

You can also generate your own dataset using the code provided in this repository.

### Example Dataset

Example of the dataset (first few rows):

```csv
feature1,feature2,feature3,feature4,target
-0.222100,0.688894,1.214023,1.557259,1
0.267051,-0.375121,-0.277932,-0.754098,0
```


