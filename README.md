# Data science competition 2022/23
## Title: **Synthetic Data Generation**
## Company: **Zurich Customer Active Management d.o.o.**

Team members:
 * `Valter Hudovernik`, `63160134`, `vh0153@student.uni-lj.si`
 * `Martin Jurkovič`, `63180015`, `mj5835@student.uni-lj.si`

Mentor: `Erik Štrumbelj`, `erik.strumbelj@fri.uni-lj.si`
***
## Project description
Synthetic relational data generation is a niche field with growing interest in the last years from the academia and
industry. We have researched the methods for generation and evaluation of synthetic tabular relational data. We
will evaluate and use the best performing model to generate data from Zurich Insurance Group. They will be able
to use this data for better ML models, faster data ingestion from their branches and easier GDPR compliance.

## Project structure
The project is divided into two parts. The first part is the research of generation and evaluation of the synthetic data. The second part is the implementation of the state of the art models and the evaluation of the generated data.

Most of the work for the research part is in `/notebooks` folder.

For file structure we follow the [cookiecutter data science template](https://drivendata.github.io/cookiecutter-data-science/).

## Building the RIKE package

To build and install the RIKE package locally run the following command in the root of the project:

```bash
pip install wheel

python setup.py sdist bdist_wheel

pip install --find-links=dist rike
```

or alternatively install in develop mode:

```bash
python setup.py develop
```