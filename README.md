# Lab 4 — Neural Networks with PyTorch

This repository contains the implementation for **Lab 4**, covering two neural network tasks using PyTorch:

* **Day 1:** Regression using the California Housing dataset
* **Day 2:** Image classification using the Fashion-MNIST dataset

The main goal of the lab is to understand what PyTorch automates compared with manually implementing gradient descent and neural network training using NumPy.

---

## Project Structure

```text
lab4_neural_networks/
│
├── day1_regression_starter.ipynb
├── day2_classification_starter.ipynb
│
├── stage_a_regression.py
├── stage_b_classification.py
├── train.py
│
├── stage_a.pt
├── stage_b.pt
│
├── requirements.txt
├── .gitignore
└── data/
```

### Files

| File                                | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `stage_a_regression.py`             | Loads and preprocesses the California Housing dataset                         |
| `stage_b_classification.py`         | Loads Fashion-MNIST and creates the train/validation/test data loaders        |
| `train.py`                          | Contains reusable training functions for regression and classification        |
| `day1_regression_starter.ipynb`     | Day 1 experiments, training, evaluation, plots, and analysis                  |
| `day2_classification_starter.ipynb` | Day 2 experiments, validation, early stopping, confusion matrix, and analysis |
| `stage_a.pt`                        | Saved PyTorch regression model                                                |
| `stage_b.pt`                        | Saved PyTorch classification model                                            |
| `requirements.txt`                  | Python dependencies                                                           |
| `data/`                             | Dataset files downloaded during execution                                     |

The `.gitignore` excludes the dataset and Python cache files from version control.

---

# Day 1 — California Housing Regression

## Objective

The first part of the lab builds a small **Multi-Layer Perceptron (MLP) regressor** using PyTorch and compares its performance against the linear regression approach implemented manually with NumPy in Lab 2.

The California Housing dataset is loaded using `sklearn.datasets.fetch_california_housing`. The data is split into training and test sets, and the input features are standardized using a `StandardScaler` fitted only on the training data. The target is reshaped to `(n, 1)` so that it matches the model output shape.

## Model

The regression model uses a small MLP with the following structure:

```text
Input Features
      ↓
Linear Layer
      ↓
ReLU
      ↓
Linear Layer
      ↓
Output
```

The model is trained using:

* **Loss:** `nn.MSELoss()`
* **Optimizer:** `optim.Adam`

The training loop performs the standard PyTorch sequence:

```text
Forward Pass
    ↓
Calculate Loss
    ↓
Zero Gradients
    ↓
Backpropagation
    ↓
Optimizer Step
```

The reusable training function is implemented in `train.py`.

## Evaluation

After training:

* The training loss is plotted.
* The final test MSE is calculated.
* The MLP is compared against the Lab 2 linear regression implementation.

Since the California Housing problem is relatively close to linear, the MLP is not expected to dramatically outperform the linear model. The comparison demonstrates the benefit of automatic differentiation and framework-based neural network training rather than simply showing that a neural network always performs better.

The trained model is saved using:

```python
torch.save(model.state_dict(), "stage_a.pt")
```

---

# Day 2 — Fashion-MNIST Classification

## Objective

The second part of the lab builds a neural network classifier for the **Fashion-MNIST** dataset.

Fashion-MNIST contains 28×28 grayscale images belonging to 10 clothing categories.

The training data is split into:

* Training set — 90%
* Validation set — 10%
* Test set — kept separate until final evaluation

The training `DataLoader` is shuffled, while the validation and test loaders are not shuffled.

## Model

The images are flattened from:

```text
28 × 28 = 784
```

pixels into a 784-dimensional input vector.

The classifier is a small MLP that produces scores for the 10 Fashion-MNIST classes.

The final layer outputs **raw class scores (logits)**. No softmax is added because `nn.CrossEntropyLoss()` handles the required log-softmax operation internally.

## Training

The classification model is trained using mini-batches.

For every epoch:

1. Train on the training batches.
2. Calculate the average training loss.
3. Evaluate the model on the validation set.
4. Calculate validation accuracy.
5. Save the best-performing model.
6. Apply early stopping if validation accuracy does not improve for the specified patience.

The implementation uses a patience-based early stopping mechanism rather than stopping after a single non-improving epoch. The best model state is cloned and restored after training.

The reusable classification training function is located in `train.py`.

## Evaluation

The final model is evaluated on the test set.

The notebook includes:

* Training loss curve
* Validation accuracy curve
* Test accuracy
* Confusion matrix
* Brief interpretation of classification errors

The expected Fashion-MNIST test accuracy is approximately **87–89%**, depending on the exact training configuration.

The confusion matrix is used to identify which clothing categories are most frequently confused, such as shirts, coats, pullovers, and T-shirts.

---

# Installation

## Requirements

The project requires:

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib

The provided `requirements.txt` specifies the required package versions.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

---

# How to Run

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd lab4_neural_networks
```

## 2. Create and activate a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Day 1

Open:

```text
day1_regression_starter.ipynb
```

Run the notebook from top to bottom.

The notebook will:

1. Load the California Housing dataset.
2. Preprocess and standardize the features.
3. Build the MLP regression model.
4. Train the model using PyTorch.
5. Plot the training loss.
6. Calculate the test MSE.
7. Compare the MLP with the Lab 2 linear regression model.
8. Save the trained model as `stage_a.pt`.

---

## 5. Run Day 2

Open:

```text
day2_classification_starter.ipynb
```

Run the notebook from top to bottom.

The notebook will:

1. Load Fashion-MNIST.
2. Create training, validation, and test loaders.
3. Display sample images.
4. Build the MLP classifier.
5. Train using mini-batches.
6. Track training loss and validation accuracy.
7. Apply early stopping.
8. Restore the best model.
9. Evaluate on the test set.
10. Display and interpret the confusion matrix.
11. Save the trained model as `stage_b.pt`.

Fashion-MNIST is downloaded automatically if it is not already present in the `data/` directory.

---

# Important Implementation Details

### Data preprocessing

For regression, `StandardScaler` is fitted only on the training data and then applied to the test data to avoid data leakage. The regression targets are reshaped to `(n, 1)`.

### Training loss

Loss values are stored using `.item()` so that computation graphs are not retained unnecessarily.

### Evaluation mode

Validation and test evaluation use:

```python
model.eval()
```

and disable gradient tracking with:

```python
torch.no_grad()
```

### Early stopping

The classification model uses a patience-based early stopping mechanism. The best model parameters are cloned rather than stored as a live reference, and the best state is restored after training.

---

# Expected Outputs

## Day 1

The notebook should produce:

* Training loss curve
* Final test MSE
* Comparison between the PyTorch MLP and Lab 2 linear regression
* Written analysis of when an MLP would decisively outperform a linear model

## Day 2

The notebook should produce:

* Example Fashion-MNIST images
* Training loss curve
* Validation accuracy curve
* Early stopping
* Final test accuracy
* Confusion matrix
* Written analysis of the training/validation gap
* Closing reflection on what PyTorch did and did not automate

These outputs correspond to the required Lab 4 deliverables.

---

# Saved Models

The trained model parameters are saved using PyTorch `state_dict`s:

```text
stage_a.pt    → Day 1 regression model
stage_b.pt    → Day 2 classification model
```

These files contain the learned model parameters rather than the complete Python model definition.

---

# Troubleshooting

### Regression loss does not decrease

Make sure `optimizer.zero_grad()` is called before `loss.backward()`.

### Regression loss becomes NaN

Check that the input features were standardized and consider reducing the learning rate.

### Regression MSE looks incorrect

Check that the target has shape:

```text
(n, 1)
```

rather than:

```text
(n,)
```

### Classification accuracy is around 10%

Check that:

* The model does not contain a final softmax.
* The labels are integer class indices.
* `CrossEntropyLoss` is being used.

### Early stopping does not work

Make sure the patience counter increases when validation accuracy does not improve and resets when a new best validation accuracy is achieved.

### Dataset download issues

Fashion-MNIST should be cached in the `data/` directory after the first download.

---

# Learning Outcomes

This lab demonstrates the transition from manually implemented NumPy machine learning algorithms to framework-based deep learning.

The main concepts covered are:

* PyTorch tensors
* Neural network modules
* Forward propagation
* Loss functions
* Automatic differentiation
* Backpropagation
* Optimizers
* Training loops
* Mini-batch training
* Validation sets
* Early stopping
* Model checkpointing
* Classification accuracy
* Confusion matrices
* Data preprocessing
* Data leakage
* Model evaluation

The lab also highlights an important practical distinction: PyTorch automates much of the gradient computation and optimization workflow, but the programmer still has to correctly design the model, prepare the data, choose the appropriate loss, manage train/evaluation modes, and evaluate the results.

---

# Submission

The final project should be pushed to a **public GitHub repository** containing at minimum:

```text
train.py
stage_a_regression.py
stage_b_classification.py
day1_regression_starter.ipynb
day2_classification_starter.ipynb
requirements.txt
README.md
.gitignore
```

The assignment instructions specify submitting the public GitHub repository link on Moodle. Before submission, the notebooks should be restarted and run from top to bottom to verify that the entire project works correctly.
