# Neural Network Weight Initialization Analysis

This repository contains a CS478 deep learning notebook that explores how neural network weight initialization affects activation outputs. The project compares Glorot normal and Glorot uniform initialization across sigmoid, tanh, and ReLU activation functions using TensorFlow, Keras, NumPy, and Matplotlib.

## Project Overview

Weight initialization is an important part of neural network design. Poor initialization can cause activations to collapse, saturate, or become unstable before training even begins. This notebook builds small dense neural network layers and visualizes the resulting activation distributions under different initialization and activation pairings.

The project uses a fixed random input vector with 784 features, matching the flattened size of a 28 by 28 image. A dense layer with 256 neurons is initialized with different strategies, then passed through different activation functions. The activation values are printed and plotted as histograms so the output behavior can be compared.

## Experiments Included

The notebook evaluates these initializer and activation combinations:

- Glorot normal with sigmoid
- Glorot normal with tanh
- Glorot normal with ReLU
- Glorot uniform with sigmoid
- Glorot uniform with tanh
- Glorot uniform with ReLU

Each experiment resets the TensorFlow random seed so the comparisons are reproducible.

## Example Results

The notebook prints summary statistics for each activation distribution.

```text
Glorot normal with sigmoid
Mean 0.5053
Standard deviation 0.1665
Minimum 0.1407
Maximum 0.9083

Glorot normal with tanh
Mean 0.0015
Standard deviation 0.5390
Minimum -0.9561
Maximum 0.9607

Glorot normal with ReLU
Mean 0.3021
Standard deviation 0.4267
Minimum 0.0000
Maximum 1.7740

Glorot uniform with sigmoid
Mean 0.5079
Standard deviation 0.1660
Minimum 0.1392
Maximum 0.8769

Glorot uniform with tanh
Mean 0.0237
Standard deviation 0.5083
Minimum -0.9474
Maximum 0.9557

Glorot uniform with ReLU
Mean 0.3049
Standard deviation 0.4230
Minimum 0.0000
Maximum 1.7662
```

## Key Takeaways

Sigmoid activations produce values mostly between 0 and 1, centered near 0.5.

Tanh activations produce values between -1 and 1 and remain centered near 0.

ReLU activations produce many zero values and positive outputs, creating a skewed distribution.

Glorot normal and Glorot uniform both keep the activations in a controlled range, but their output distributions vary slightly.

## Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow
- Keras
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys_CS478_01_CA5_Weight_Initialization.ipynb
├── Jason_Stys_CS478-01_CA5_Weight_Initialization.pdf
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/neural-network-weight-initialization-analysis.git
cd neural-network-weight-initialization-analysis
```

Install the required packages.

```bash
pip install numpy matplotlib tensorflow notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- TensorFlow and Keras model construction
- Neural network weight initialization
- Dense layer activation analysis
- Reproducible experiment setup with random seeds
- Activation distribution visualization
- NumPy based input generation
- Matplotlib histogram plotting
- Jupyter Notebook workflow

## Possible Future Improvements

- Add He initialization for ReLU comparison
- Add LeCun initialization for sigmoid and tanh comparison
- Compare deeper networks across several layers
- Track activation variance layer by layer
- Add gradient distribution plots
- Convert the experiment into a reusable Python script
