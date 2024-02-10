# ELAAN: Additive Models with Sparse and Structured-Interactions

This is our implementation of End-to-End Learning Approach for Additive Models with interactions under sparsity as described in our manuscript.

[Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions](http://arxiv.org/abs/2108.11328) by Shibal Ibrahim, Peter Radchenko, Emanuel Ben-David, Rahul Mazumder

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
* `ELAAN-I`: Additive Models with Interactions under L0
* `ELAAN-H`: Additive Models with Interactions with Strong Hierarchy

## Running Code
Scripts folder contains different bash scripts for running ELAAN-I, ELAAN-H, EBM and GamiNet on Census data as well as Synthetic datasets for different seeds.

For example, ELAAN-I can be run for one seed on Census data as follows:
```bash
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaani/elaani_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed 1 --relative_penalty 1.0 --grid_search 'reduced' --run_first_round --version 1 --eval_criteria 'mse' --logging
```

Similarly, GamiNet can be run as: 
```bash
cd /home/gridsan/shibal/elaan/baselines/GamiNet/examples/

/home/gridsan/shibal/.conda/envs/additive2/bin/python /home/gridsan/shibal/elaan/baselines/GamiNet/examples/gaminet_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed 1 --version 1
```
Note GamiNet requires additional installation of tensorflow and tensorflow-lattice libraries 


ELAAN-H can be run on synthetic data as follows:
```bash
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaanh/elaanh_synthetic.py  --dataset 'synthetic' --dist 'normal' --correlation 0.5 --seed $SLURM_ARRAY_TASK_ID --train_size 100 --version 1 --r 1.0 --Ki 10 --Kij 5
```
Note ELAAN-H requires additional installation of gurobipy, as we use gurobi to solve the convex relaxation.

For Census data, we use ELAAN-I path of solutions to generate a reduced support, which can be used by ELAAN-H to solve problem under strong hierarchy. Hence, this requires ELAAN-I script to have finished first to generate some model files, which are loaded. 


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


