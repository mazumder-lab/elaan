# ELAAN: Additive Models with Sparse and Structured-Interactions

This is our implementation of End-to-End Learning Approach for Additive Models with interactions under sparsity as described in our manuscript.

[Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions](http://arxiv.org/abs/2108.11328) by Shibal Ibrahim, Rahul Mazumder, Peter Radchenko, Emanuel Ben-David

## Installation
We provide a conda environment file named "sparse-am-with-interactions.yml" for straightforward installation with Anaconda, which can be used to setup a sparse-am-with-interactions environment with the commands:

conda env create --name elaan --file=elaan.yml

source activate elaan

Alternatively, the following packages can be downloaded to run the python scripts and jupyter notebooks.

## Requirements
* descartes                 1.1.0
* dill                      0.3.3 
* fiona                     1.8.18
* gurobi                    9.0.1 
* ipywidgets                7.5.1
* matplotlib                3.3.2 
* notebook                  6.1.5
* numpy                     1.19.4 
* pandas                    1.1.4
* patsy                     0.5.1
* pyproj                    2.6.1.post1
* python                    3.6.12 
* rtree                     0.9.4
* scikit-learn              0.23.2
* scipy                     1.5.3
* tqdm                      4.54.1
 
## Proposed Models
* `ELAAN`: Additive Models with only main effects only under L0
* `ELAAN-I`: Additive Models with Interactions under L0
* `ELAAN-H`: Additive Models with Interactions with Strong Hierarchy

## Running the code

```bash
cd src

The following 2 scripts can be used to run the following two models on Census Data:
For ELAAN-I: run elaani/elaani-census.py
For ELAAN-H: run elaanh/elaanh-census.py
```

## Citing Additive-Models-with-Structured-Interactions
If you find our repository useful in your research, please consider citing the following paper.

```
@article{Ibrahim2021,
      title={Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions}, 
      author={Shibal Ibrahim and Rahul Mazumder and Peter Radchenko and Emanuel Ben-David},
      year={2021},
      volume={abs/2108.11328},
      journal={arXiv},
}
```


