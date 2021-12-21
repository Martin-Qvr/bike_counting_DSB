from pathlib import Path
from xgboost import XGBRegressor
import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

def confinement(X):

    date = pd.to_datetime(X['date'])
    X.loc[:, ['date_only']] = date
    new_date = [dt.date() for dt in X['date_only']]
    X.loc[:, ['date_only']] = new_date
    mask = ((X['date_only'] >= pd.to_datetime('2020/10/30'))
        & (X['date_only'] <= pd.to_datetime('2020/12/15'))
        | (X['date_only'] >= pd.to_datetime('2021/04/03'))
        & (X['date_only'] <= pd.to_datetime('2021/05/03')))
    X['confi'] =np.where(mask, 1, 0)
    return X.drop(columns=['date_only'])

def curfew(X):
    date = pd.to_datetime(X['date'])
    X.loc[:, ['date_only']] = date
    new_date = [dt.date() for dt in X['date_only']]
    X.loc[:, ['date_only']] = new_date
    X.loc[:, ['hour_only']] = date
    new_hour = [dt.hour for dt in X['hour_only']]
    X.loc[:, ['hour_only']] = new_hour
    mask = (
        #First curfew
        (X['date_only'] >= pd.to_datetime('2020/12/15'))
        & (X['date_only'] < pd.to_datetime('2021/01/16'))
        & ((X['hour_only'] >= 20) | (X['hour_only'] <= 6))

        | 
        
        # Second curfew
        (X['date_only'] >= pd.to_datetime('2021/01/16'))
        & (X['date_only'] < pd.to_datetime('2021/03/20'))
        & ((X['hour_only'] >= 18) | (X['hour_only'] <= 6))

        |

        # Third curfew
        (X['date_only'] >= pd.to_datetime('2021/03/20'))
        & (X['date_only'] < pd.to_datetime('2021/05/19'))
        & ((X['hour_only'] >= 19) | (X['hour_only'] <= 6))

        |

        # Fourth curfew
        (X['date_only'] >= pd.to_datetime('2021/05/19'))
        & (X['date_only'] < pd.to_datetime('2021/06/9'))
        & ((X['hour_only'] >= 21) | (X['hour_only'] <= 6))

        |

        # Fifth curfew
        (X['date_only'] >= pd.to_datetime('2021/06/9'))
        & (X['date_only'] < pd.to_datetime('2021/06/20'))
        & ((X['hour_only'] >= 21) | (X['hour_only'] <= 6))
        )
    X['curfew'] =np.where(mask, 1, 0)

    return X.drop(columns=['hour_only', 'date_only'])

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't','u']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day', 'weekday']
    numeric_cols = ['t','u']
    hour_col = ['hour']
    period_cols = ['confi','curfew']

    categorical_cols = ["counter_name"]


    preprocessor = ColumnTransformer([
        ('date', StandardScaler(), date_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('numeric', StandardScaler(), numeric_cols),
        ('period', 'passthrough', period_cols),
        ('hour', PolynomialFeatures(degree=2), hour_col)
        ]
    )

    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=9,  
        min_child_weight=0.001, 
        gamma=0.1, 
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=100,
        learning_rate=0.2,
        )

    pipe = make_pipeline(FunctionTransformer(confinement, validate=False),FunctionTransformer(curfew, validate=False),FunctionTransformer(_merge_external_data, validate=False), date_encoder, preprocessor, regressor)

    return pipe
