from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'RIKE - A Python package for evaluating synthetic relational datasets'
LONG_DESCRIPTION = 'RIKE - A Python package for evaluating synthetic relational datasets'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="rike",
    version=VERSION,
    author="Martin Jurkovic, Valter Hudovernik",
    author_email="martin.jurkovic19@gmail.com, valter.hudovernik@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        "matplotlib==3.7.1",
        "numpy==1.24.2",
        "pandas==1.5.3",
        "REaLTabFormer==0.1.1",
        "scikit-learn==1.2.2",
        "scipy==1.10.1",
        "sdmetrics==0.9.2",
        "sdv==0.18.0",
        "tqdm==4.65.0",],
    keywords=['python', 'RIKE', 'synthetic data', 'relational data', 'evaluation'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
