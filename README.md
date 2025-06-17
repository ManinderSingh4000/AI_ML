# AI_ML

This repository contains machine learning projects and tools, with a primary focus on a Streamlit-based application for student placement prediction and analytics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Requirements](#requirements)
- [Notebooks & Additional Materials](#notebooks--additional-materials)
- [License](#license)

## Overview

The main feature of this repository is a web application built with Streamlit that allows users to:
- Explore student placement data.
- Add new student profiles.
- Predict the placement probability of a student based on CGPA, internships, and projects using a machine learning model (Random Forest Classifier).

## Features

- **Interactive Dashboard:** Visualize placement statistics, CGPA distribution, and student data.
- **Add Student:** Easily input new student records.
- **Placement Prediction:** Predict the likelihood of a student being placed using a trained machine learning model.
- **Jupyter Notebooks:** Includes model-building and experimentation notebooks.
- **Reference Materials:** Contains a PDF on transformer models for further study.

## Project Structure

```
.
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Python dependencies
├── Guit_model.ipynb        # Jupyter Notebook (modeling/experiments)
├── Model_G_Model.ipynb     # Jupyter Notebook (modeling/experiments)
├── Spec.ipynb              # Jupyter Notebook (specifications/results)
├── guit_model.py           # Python script (model-related code)
├── model_g_model.py        # Python script (model-related code)
├── transformers.pdf        # Reference PDF on transformer models
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ManinderSingh4000/AI_ML.git
cd AI_ML
```

### 2. Install Requirements

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

- **Dashboard:** View all student data and placement statistics.
- **Add Student:** Add new students via the sidebar and update the dataset.
- **Predict Placement:** Enter student details to get a placement prediction.

## Requirements

- streamlit
- pandas
- numpy
- scikit-learn

(See `requirements.txt` for the full list.)

## Notebooks & Additional Materials

- `Guit_model.ipynb`, `Model_G_Model.ipynb`, `Spec.ipynb`: Jupyter notebooks for model development, experiments, and specifications.
- `guit_model.py`, `model_g_model.py`: Python scripts for modeling.
- `transformers.pdf`: Reference material on transformer architectures.

## License

This project is released under the MIT License.

---

*Created by [ManinderSingh4000](https://github.com/ManinderSingh4000)*
