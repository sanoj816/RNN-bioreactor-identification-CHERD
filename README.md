# RNN-bioreactor-identification-CHERD
# Source Code for: Recurrent neural network-based multivariable identification of a nonlinear continuous ethanol fermentation bioreactor

## Overview

This repository contains the source code and plotting utilities accompanying the manuscript submitted for publication.

The provided implementation demonstrates the workflow used for nonlinear dynamic identification of a bioreactor using recurrent neural networks (RNNs), including:

- Data generation using the nonlinear bioreactor model
- Dataset preparation
- Training of recurrent neural network models
- Model evaluation
- Free-run forecasting
- Noise robustness analysis
- Export of results for visualization

The repository is intended to support scientific transparency and allow readers to understand the computational workflow described in the manuscript.

---

## Repository Structure

```
.
├── CHERD_upload_review.py      # Main Python script
├── CHERD_plot.mlx              # MATLAB Live Script for generating figures
├── results/                    # Automatically generated after execution
│   ├── data/
│   └── models/
└── README.md
```

---

## Requirements

### Python

The code was developed using Python 3.x with the following major packages:

- NumPy
- Pandas
- SciPy
- scikit-learn
- TensorFlow / Keras
- Joblib

Install the required packages

---

### MATLAB

MATLAB (R2023a or later recommended)

The supplied Live Script

```
CHERD_plot.mlx
```

is used only to generate the publication-quality figures from the data produced by the Python script.

---

## Usage

### Step 1

Run

```
CHERD_upload_review.py
```

This script

- generates the simulation dataset,
- trains the recurrent neural network models,
- evaluates prediction performance,
- performs forecasting and robustness analyses,
- saves the generated models and output data.

Depending on the available hardware, execution may require a considerable amount of time.

---

### Step 2

After the Python script completes successfully,

open

```
CHERD_plot.mlx
```

in MATLAB.

Execute the Live Script to reproduce the figures presented in the manuscript from the generated data.

---

## Output

The Python program automatically creates a directory named

```
results/
```

containing the generated models and intermediate data required for post-processing and visualization.

---

## Notes

- The supplied implementation is intended to accompany the published manuscript.
- The plotting workflow has been separated from the model development workflow to improve readability and reproducibility.
- Depending on the computing platform, execution time may vary.



