# Synthetic Data Generation Project

## Overview

Synthetic relational data generation is a niche field with growing interest in recent years, both in academia and industry. We have conducted research on methods for generating and evaluating synthetic tabular relational data. The goal is to identify the best-performing model and use it to generate data for Zurich Insurance Group. This generated data will help improve their machine learning models, facilitate faster data ingestion from branches, and ensure easier compliance with GDPR regulations.


## Team

- Valter Hudovernik (63160134, vh0153@student.uni-lj.si)
- Martin Jurkovič (63180015, mj5835@student.uni-lj.si)

## Mentor

Erik Štrumbelj (erik.strumbelj@fri.uni-lj.si)

## Project Description

This project focuses on the generation and evaluation of synthetic relational data. The goal is to explore and evaluate state-of-the-art models for generating synthetic data, which can be utilized by Zurich Insurance Group. Synthetic data has the potential to streamline data ingestion processes, facilitate GDPR compliance and potentially enhance machine learning models.

## Project Structure

Here's an overview of the key directories and files:

- `README.md`: Project overview and instructions (you're currently reading this file).
- `Results.md`: Document showcasing the results and findings of the project.
- `data/`: Directory containing the project's datasets, including original, synthetic, and k-fold split data
- `models/`: Directory for storing trained models and related metadata.
- `notebooks/`: Contains Jupyter notebooks used for research, model evaluation, and data visualization.
- `src/`: Source code directory, organized into subdirectories based on functionality (data, download_scripts, evaluation_scripts, generation_scripts, rike, sling_scripts, split_scripts, etc.).
- `presentation/`: Directory for project presentation materials.
- `figs/`: Directory for storing project-related figures and visualizations.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```


Feel free to explore the repository and reach out to the team members or the mentor for any questions or inquiries.

## Additional Instructions

### Downloading benchmark datasets:
Benchmark datasets can be downloaded by running:
`src/download_scripts/download_datasets.sh`.

### Splitting data:
Scripts for splitting the downloaded datasets can be found in `src/split_scripts`. The split datasets into k folds will be saved into `data/splits/<dataset_name>`.

### Generating data:
Scripts for generating data can be found in the `src/generation_scripts` directory.  
Since RCTGAN method's dependencies clash with other methods, we provide a separate `requirements_rctgan.txt` and `setup_rctgan.py` for building the rike package with RCTGAN. The synthetic datasets split into k folds will be saved into `data/splits/<dataset_name>`.

### Evaluating data:
To evaluate the synthetic datasets, run `src/evaluation_scripts/benchmark.py`.  
Reports of the metrics will be saved into `metrics_report/` as `.json` files. In the `metrics_report/` directory, you will also find all of the results produced during this project.

### Results:
Visualizations of all the results from the project are in [RESULTS.md](./RESULTS.md).  
The final report is available [here](./final_report/report.pdf).
