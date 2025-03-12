import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna

epsilon=1e-4
train_data = pd.read_csv('data/train_data.csv')
train_data = train_data.loc[train_data['Target_Variable/Total Income'] != 0].copy()

train_data['Target_Variable/Total Income'] = np.log((train_data['Target_Variable/Total Income']+epsilon).clip(lower=1e-8))

# Separate raw features and target
X_raw = train_data.drop(columns=['Target_Variable/Total Income'])
y = train_data['Target_Variable/Total Income']

def objective(trial):
    mapes=[]
    params = {
        'objective': 'reg:squarederror',  
        'eval_metric': 'rmse',
        'learning_rate': 0.075,
        'max_depth': 10,
        'subsample': 0.3,
        'colsample_bytree': 0.7,
        'gamma': 7,
        'reg_alpha': 5,
        'reg_lambda': 3,
        'nthread':-1,
        'seed': 37                     
    }

    kf = KFold(n_splits=7, shuffle=True, random_state=69)

    fold = 1
    for train_idx, val_idx in kf.split(X_raw):
        
        X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        num_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        
        ct = ColumnTransformer(
            [
                ('num', MinMaxScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ],
            remainder='passthrough'
        )
        
        X_train = ct.fit_transform(X_train_raw)
        X_val = ct.transform(X_val_raw)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        preds_log = model.predict(dval)           
        preds_exp = np.exp(preds_log)            
        y_val_exp = np.exp(y_val) - epsilon                

        mape = mean_absolute_percentage_error(y_val_exp, preds_exp)
        print(f'Fold {fold} MAPE: {mape:.4f}')
        mapes.append(mape)
        fold += 1
    return np.max(mapes)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# best_trial = study.best_trial
# print("Value (MAPE):", best_trial.value)
# print("Params: ")
# for key, value in best_trial.params.items():
#     print(f"{key}: {value}")

objective(49385)