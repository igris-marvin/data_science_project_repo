# GIT REPOSITORY LINK
https://github.com/igris-marvin/data_science_project_repo

# Water Supply Reliability Risk Predictor

This project provides a machine learning–powered Streamlit web application that predicts the risk level of water supply reliability for South African Water Services Authorities (WSAs).  
It uses a trained Deep Tabular Neural Network pipeline with custom feature engineering (`WSAFeatureEngineer`).

---

## 🖥️ Prerequisites

### 1. Install Python
- Download the latest stable version of Python (3.10+ recommended) from:  
  [https://www.python.org/downloads/](https://www.python.org/downloads/)
- During installation, check **"Add Python to PATH"**.

Verify installation:
```bash
python --version
```
---

2. Install VS Code
Download Visual Studio Code from:
https://code.visualstudio.com/

Install the Python extension (Python Extension Pack - by Don Jayamanne) for VS Code to enable notebook/script execution.

## 🚀 Deployment Instructions
Steps:

Open the root folder of the project in vscode

Open CMD in the root folder in vscode

### 1. Environment Setup
```bash

# setup python environment
python -m venv .venv

# install relevant dependencies
pip install <>

# Activate (PowerShell) - this activates and runs the python virtual enviroment
.venv\Scripts\Activate.ps1

# on the activated python environment in vscode integrated cmd, type the following command to lauch the app
streamlit run app.py

```

<!-- * -->
## 🧪 Demonstration Cases

### Case 1 – Medium Risk
- Population: 1000000  
- Access to Water: 800009 (80%)  
- Urban Pop: 120000, Urban Access: 15000 (12.5%)  
- Rural Pop: 300000, Rural Access: 300000 (100%)  
Output: MEDIUM RISK (100% confidence)

---

### Case 2 – Low Risk (Metro Municipality)
- Population: 2500000  
- Access to Water: 2450000 (98%)  
- Urban Pop: 2000000, Urban Access: 1980000 (99%)  
- Rural Pop: 500000, Rural Access: 470000 (94%)  
Output: LOW RISK (~96.5% confidence)

---

### Case 3 – High Risk (Rural District)
- Population: 450000  
- Access to Water: 180000 (40%)  
- Urban Pop: 50000, Urban Access: 30000 (60%)  
- Rural Pop: 400000, Rural Access: 150000 (37.5%)  
Output: HIGH RISK (~92.8% confidence)

---

### Case 4 – Invalid Input
- Population: 100000  
- Access to Water: 150000 (exceeds total)  
- Urban Pop: 60000, Urban Access: 80000 (exceeds urban)  
Validation Error:  
“Population with access cannot exceed Total Population.  
Urban Population with access cannot exceed Urban Population.”


<!-- ^ IMPORTANT NOTE -->
The dataset csv was edited to remove double space into single space in 2 columns to avoid errors during runtime like exact names 