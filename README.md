# Dependency Triad

Artifact for the paper **“Dependency Triad: A Metric to Quantify Dependencies Between Attributes for Local Differential Privacy.”**

**License:** Apache License 2.0

## 1. Overview and Scope

This artifact provides the implementation of the **Dependency Triad (DT)** metric proposed in the paper. It is not intended to reproduce every figure or full-scale experiment reported in the paper. Instead, it focuses on the core implementation of DT and demonstrates how the DT parameters $(\alpha, \beta, \delta)$ can be computed and used to estimate correlation-induced privacy leakage (CPL).


## 2. Repository Structure

The repository is organized as follows.

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── Calibrate/
│   │   └── calibrate.py
│   └── CPL_metrics/
│       ├── DT.py
│       ├── LTM.py
│       ├── LTM_pytorch.py
│       ├── HCC-1.py
│       ├── HCC-1_pytorch.py
│       ├── HCC-2.py
│       ├── calibrate_CPL.py
│       └── uncalibrate_CPL.py
├── datasets/
│   └── synthetic/
├── results/
│   └── synthetic/
└── test.ipynb
```

## 3. Main Component: DT

The main implementation of DT is provided in:

```text
src/privacy_leakage/DT.py
```

This file implements Algorithm 1 and Algorithm 2 from the paper.

### Main Function

```python
DT(eps, CMF, Delta=0, leakage_only=False, DT_parameters=(0, 0, 0), get_parameters=False)
```

### Parameters

- `eps` (`float`):  
  The local differential privacy budget.

- `CMF` (`numpy.ndarray`):  
  The conditional probability matrix representing \(p(\hat{x} \mid x)\), where each row corresponds to a value of \(X_k\), and each column corresponds to a value of \(\hat{X}\).

- `Delta` (`numpy.ndarray` or `int`, optional):  
  The distributional uncertainty matrix associated with `CMF`. If `Delta=0`, no distributional uncertainty is assumed.  
  Default: `0`.

- `leakage_only` (`bool`, optional):  
  If `True`, the function skips the computation of DT parameters and directly estimates CPL using the provided `DT_parameters`.  
  Default: `False`.

- `DT_parameters` (`tuple`, optional):  
  A tuple containing previously computed DT parameters \((\alpha, \beta, \delta)\). This is used only when `leakage_only=True`.  
  Default: `(0, 0, 0)`.

- `get_parameters` (`bool`, optional):  
  If `True`, the function returns the computed DT parameters \((\alpha, \beta, \delta)\) instead of the estimated CPL.  
  Default: `False`.

### Returns

- If `get_parameters=True`:  
  Returns the DT parameters:

```python
(alpha, beta, delta)
```

- Otherwise:  
  Returns the estimated correlation-induced privacy leakage:

```python
CPL
```

### Example Usage

```python
from privacy_leakage.DT import DT

# Compute DT parameters
alpha, beta, delta = DT(
    eps=1.0,
    CMF=CMF,
    Delta=Delta,
    get_parameters=True
)

# Estimate CPL using computed DT parameters
cpl = DT(
    eps=1.0,
    CMF=CMF,
    Delta=Delta
)

# Estimate CPL directly from previously computed DT parameters
cpl = DT(
    eps=1.0,
    leakage_only=True,
    DT_parameters=(alpha, beta, delta)
)
```

## 4. Software Requirements

The artifact was developed for Python 3.

Required packages include:

```text
numpy
scipy
matplotlib
torch
pandas
seaborn
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/SandaruJayawardana/CoP-artifacts-PETS
cd CoP-artifacts-PETS
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Running the Smoke Test

Run $test.ipynb$.

Expected output:

Two plots that depict the estimated CPLs and estimation error (Figure 4(a) and 4(b) in the paper).

## 7. License

The source code is released under the Apache License 2.0.

Datasets are subject to their original licenses. 

## 8. Contact

For questions about this artifact, contact:

```text
Sandaru Jayawardana (sjay9734@uni.sydney.edu.au)
```
