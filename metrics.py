import pandas as pd, numpy as np
def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        mape = np.mean(np.abs((y_true - y_pred) / y_true)[mask])
        return 0 if np.isnan(mape) else float(mape)
    except:
        return 0
        
def SMAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (abs(y_true) + abs(y_pred) != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        nominator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean((2.0 * nominator / denominator)[mask])
        return 0 if np.isnan(smape) else float(smape)
    except:
        return 0

def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mse = ((y_true - y_pred) ** 2)[mask].mean()
        return 0 if np.isnan(mse) else float(mse)
    except:
        return 0

def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return float(rmse)

def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mae = abs(y_true - y_pred)[mask].mean()
        return 0 if np.isnan(mae) else float(mae)
    except:
        return 0

def confusion_matrix(act, pred):
    predtrans = ['Up' if i > 0.5 else 'Down' for i in pred]
    actuals = ['Up' if i > 0 else 'Down' for i in act]
    confusion_matrix = pd.crosstab(pd.Series(actuals), pd.Series(predtrans),
                                   rownames = ['Actual'], colnames = ['Predicted'])
    return confusion_matrix