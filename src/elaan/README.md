## ELAAN: Additive Models with main effects under L0

This folder contains the source files for running the additive model under L0 (Only main effects)

The following files serve the following purpose:
- models.py (defines the ELAAN object which is called by the elaan_census.py). 
- ValidationPath.py (Hyperparameter grid search with warm-starts over smoothness penalty: lambda_1). 
- L0path.py (Hyperparameter grid search with warm-starts over L0 regularization lambda_2 for each value of lambda_1).
- CoordinateDescent.py (Functions for running cyclic block coordinate descent over the covariate set)
- utilities.py (B-Spline generation and quadratic penalties generation)
