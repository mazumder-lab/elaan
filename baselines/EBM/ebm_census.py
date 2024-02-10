import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.
from contextlib import redirect_stdout

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

import optuna
from optuna.samplers import RandomSampler
import timeit

import argparse

def metrics(y, ypred, y_preprocessor=None):
    """Evaluates metrics.
    
    Args:
        y:
        ypred:
        y_scaler:
        
    Returns:
        mse: mean squared error, float scaler.
        rmse: root mean squared error, float scaler.
    """
    if y_preprocessor is not None:
        y = y_preprocessor.inverse_transform(y)
        ypred = y_preprocessor.inverse_transform(ypred)
    
    mse = mean_squared_error(y, ypred)
    rmse = mean_squared_error(y, ypred, squared=False)
    return mse, rmse

## Load data
def load_data(load_directory='./',
              filename='pdb2019trv3_us.csv',
              remove_margin_of_error_variables=False): 
    """Loads Census data, and retrieves covariates and responses.
    
    Args:
        load_directory: Data directory for loading Census file, str.
        filename: file to load, default is 'pdb2019trv3_us.csv'.
        remove_margin_of_error_variables: whether to remove margin of error variables, bool scaler.
        
    Returns:
        df_X, covariates, pandas dataframe.
        df_y, target response, pandas dataframe.
    """
    file = os.path.join(load_directory, filename)
    df = pd.read_csv(file, encoding = "ISO-8859-1")
    df = df.set_index('GIDTR')
    
    # Drop location variables
    drop_location_variables = ['State', 'State_name', 'County', 'County_name', 'Tract', 'Flag', 'AIAN_LAND']
    df = df.drop(drop_location_variables, axis=1)
    
    target_response = 'Self_Response_Rate_ACS_13_17'
    # Remove extra response variables 
    # Remove response columns 'FRST_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the first form mailed was completed and returned) and 'RPLCMNT_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the replacement form was completed and returned)

    extra_response_variables = [
        'Census_Mail_Returns_CEN_2010',
        'Mail_Return_Rate_CEN_2010',
        'pct_Census_Mail_Returns_CEN_2010',
        'Low_Response_Score',
        'Self_Response_Rate_ACSMOE_13_17',
        'BILQ_Frms_CEN_2010',
        'FRST_FRMS_CEN_2010',
        'RPLCMNT_FRMS_CEN_2010',
        'pct_FRST_FRMS_CEN_2010',
        'pct_RPLCMNT_FRMS_CEN_2010']
    df = df.drop(extra_response_variables, axis=1)
    
    if remove_margin_of_error_variables:
        df = df[np.array([c for c in df.columns if 'MOE' not in c])]

    # Change types of covariate columns with dollar signs in their values e.g. income, housing price  
    df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].replace('[\$,]', '', regex=True).astype(np.float64)

    # Remove entries with missing predictions
    df_full = df.copy()
    df = df.dropna(subset=[target_response])

    df_y = df[[target_response]]
    df_X = df.drop([target_response], axis=1)


    return df_X, df_y, df_full

def process_data(df_X,
                 df_y,
                 val_ratio=0.1, 
                 test_ratio=0.1, 
                 seed=None,
                 standardize_response=False):
    """Preprocesses covariates and response and generates training, validation and testing sets.
    
      Features are processed as follows:
      Missing values are imputed using the mean. After imputation, all features are standardized. 

      Responses are processed as follow:
      Either standardized or not depending on user choice selected by standardize_response.

    Args:
        val_ratio: Percentage of samples to be used for validation, float scalar.
        test_ratio: Percentage of samples to be used for testing, float scalar.
        seed: for reproducibility of results, int scalar.
        standardize_response: whether to standardize target response or not, bool scalar.
        
    Returns:
        X_train: Training processed covariates, float numpy array of shape (N, p).
        y_train: Training (processed) responses, float numpy array of shape (N, ).
        X_val: Validation processed covariates, float numpy array of shape (Nval, p).
        y_val: Validation (processed) responses, float numpy array of shape (N, ).
        X_test: Test processed covariates, float numpy array of shape (Ntest, p).
        y_test: Test (processed) responses, float numpy array of shape (N, ).
        x_preprocessor: processor for covariates, sklearn transformer.
        y_preprocessor: processor for responses, sklearn transformer.
    """        
#     house_values_variables = []
#     for i in df_X.columns:
#         if 'House_Value' in i:
#             house_values_variables.append(i)        
#     df_X[house_values_variables] = 100*(df_X[house_values_variables]/(df_X[house_values_variables].max(axis=0)))
    
#     income_variables = []
#     for i in df_X.columns:
#         if 'INC' in i or 'Inc' in i:
#             income_variables.append(i)        
#     df_X[income_variables] = 100*(df_X[income_variables]/(df_X[income_variables].max(axis=0)))
        
    N, p = df_X.shape
    df_X_temp, df_X_test, df_y_temp, df_y_test = train_test_split(df_X, df_y, test_size=int(test_ratio*N), random_state=seed)
    df_X_train, df_X_val, df_y_train, df_y_val = train_test_split(df_X_temp, df_y_temp, test_size=int(val_ratio*N), random_state=seed)
    
    print("Number of training samples:", df_X_train.shape[0])
    print("Number of validation samples:", df_X_val.shape[0])
    print("Number of test samples:", df_X_test.shape[0])
    print("Number of covariates:", p)
        
    ''' Processing Covariates '''    
    continuous_features = df_X.columns
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    x_preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features)])

    X_train = x_preprocessor.fit_transform(df_X_train)
    X_val = x_preprocessor.transform(df_X_val)
    X_test = x_preprocessor.transform(df_X_test)
    
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)    
    X_train = np.round(X_train, decimals=6)
    X_val = np.round(X_val, decimals=6)
    X_test = np.round(X_test, decimals=6)
    
    ''' Processing Target Responses '''
    if standardize_response:
        y_preprocessor = StandardScaler()
    else:
        def identity_func(x):
            return np.array(x)
        y_preprocessor = FunctionTransformer(lambda x: np.array(x)) # acts as identity

    y_train = y_preprocessor.fit_transform(df_y_train)
    y_val = y_preprocessor.transform(df_y_val)
    y_test = y_preprocessor.transform(df_y_test)
                
    return X_train, y_train, X_val, y_val, X_test, y_test, (x_preprocessor, x_scaler), y_preprocessor


parser = argparse.ArgumentParser(description='EBM on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--ntrials', dest='ntrials',  type=int, default=500)

# Logging Arguments
parser.add_argument('--path', dest='path',  type=str, default='/home/gridsan/shibal')
parser.add_argument('--version', dest='version',  type=int, default=1)

args = parser.parse_args()


load_directory=args.load_directory
save_directory = os.path.join(args.path, "elaan/baselines/EBM", "results", "Census", "v{}/seed{}".format(args.version, args.seed))
os.makedirs(save_directory, exist_ok=True)

df_X, df_y, _ = load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
seed = args.seed
np.random.seed(seed)
X, Y, Xval, Yval, Xtest, Ytest, _, y_scaler = process_data(
    df_X,
    df_y,
    val_ratio=0.1, 
    test_ratio=0.1,
    seed=seed,
    standardize_response=False)

def objective(trial, X, Y, Xval, Yval, Xtest, Ytest):
    
    num_of_interactions = trial.suggest_int('num_of_interactions', 10, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    
    model = ExplainableBoostingRegressor(
        feature_names=None,
        feature_types=None,
        max_bins=256,
        max_interaction_bins=32,
        binning='quantile',
        mains='all',
        interactions=num_of_interactions, # 10
        outer_bags=8,
        inner_bags=0,
        learning_rate=learning_rate,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=0.0001,
        max_rounds=5000,
        min_samples_leaf=2,
        max_leaves=3,
        n_jobs=-2,
        random_state=42,
    )
    
    model.fit(X, Y)
    mse_train, rmse_train = metrics(Y, model.predict(X))
    mse_valid, rmse_valid = metrics(Yval, model.predict(Xval))
    mse_test, rmse_test = metrics(Ytest, model.predict(Xtest))
    print("Train: MSE:", mse_train)
    print("Val: MSE:", mse_valid)
    print("Test: MSE:", mse_test)   

    trial.set_user_attr("mse_train", mse_train)
    trial.set_user_attr("mse_valid", mse_valid)
    trial.set_user_attr("mse_test", mse_test)

    trial.set_user_attr("rmse_train", rmse_train)
    trial.set_user_attr("rmse_valid", rmse_valid)
    trial.set_user_attr("rmse_test", rmse_test)
    
    split_names = []
    for name in model.feature_names:
        split_names.append(name.split(' x '))
    main_effects = [name for name in split_names if len(name)==1]
    main_effects = np.array([(int)(name[0].split("feature_")[1]) for name in main_effects])
    interaction_effects = [name for name in split_names if len(name)==2]
    interaction_effects = np.array([[(int)(name[0].split("feature_")[1]), (int)(name[1].split("feature_")[1])] for name in interaction_effects])
    trial.set_user_attr("nnz(M)", len(main_effects))
    trial.set_user_attr("nnz(I)", len(interaction_effects))
    
    return mse_valid

start = timeit.default_timer()

study = optuna.create_study(sampler=RandomSampler(seed=0), direction='minimize')
objective_wrapper = lambda trial: objective(trial, X, Y, Xval, Yval, Xtest, Ytest)
study.optimize(objective_wrapper, n_trials=args.ntrials)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
best_trial = study.best_trial

print("  Value: {}".format(best_trial.value))

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

stop = timeit.default_timer()
hours, rem = divmod(stop-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(save_directory, "Tuning.txt"), "a") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} hyperparameter settings.\n".format(int(hours), int(minutes), seconds, args.ntrials))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds, args.ntrials)) 

df_study = study.trials_dataframe()

df_study = df_study.sort_values(by=["user_attrs_rmse_valid"])
df_study.to_csv(os.path.join(save_directory, 'study.csv'), header=True, sep='\t', encoding='utf-8', index=True)
df_study = df_study.loc[0]

mse_train = df_study["user_attrs_mse_train"]
mse_valid = df_study["user_attrs_mse_valid"]
mse_test = df_study["user_attrs_mse_test"]
rmse_train = df_study["user_attrs_rmse_train"]
rmse_valid = df_study["user_attrs_rmse_valid"]
rmse_test = df_study["user_attrs_rmse_test"]

pen = np.array(['EBM'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M)})
df = df.set_index('')


print("Train: RMSE:", rmse_train)
print("Val: RMSE:", rmse_valid)
print("Test: RMSE:", rmse_test)

df.loc['EBM','Training MSE'] = mse_train
df.loc['EBM','Validation MSE'] = mse_valid
df.loc['EBM','Test MSE'] = mse_test
df.loc['EBM','Test RMSE'] = rmse_test
df.loc['EBM','Nonzeros']=X.shape[1]
df.loc['EBM','MainEffects']=X.shape[1]
df.loc['EBM','InteractionEffects']=df_study["params_num_of_interactions"]

with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.loc[['EBM'],:].to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)