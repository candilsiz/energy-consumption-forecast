# DATA SET
# Link ---> https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error

plt.style.use("fivethirtyeight")

# StackedHybrid
class StackedHybrid:
    """
    inputs: model_1, model_1, X_1, X_2
    model_1 := weak learner
    model_2 := strong learner
    X_1 := Dataframe only includes time series features
    X_2 := Dataframe  includes all time series features
    """
    def __init__(self, model_1, model_2):
        self.model_1 = model_1  # Linear Regression
        self.model_2 = model_2  # XGBoost
        
    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_pred_1 = self.model_1.predict(X_1)

        X_2_augmented = X_2.copy()
        X_2_augmented['LR_pred'] = y_pred_1  # append as new feature / stacking here
        
        X_2_train, X_2_val, y_train, y_val = train_test_split(X_2_augmented, y, test_size=0.2, shuffle=False)
        self.model_2.fit(X_2_train, y_train,
                         eval_set=[(X_2_val, y_val)],
                         eval_metric="rmse",
                         verbose=True)
        
    def predict(self, X_1, X_2):
        y_pred_1 = self.model_1.predict(X_1)
        
        X_2_augmented = X_2.copy()
        X_2_augmented['LR_pred'] = y_pred_1
        
        X_2_train, X_2_val, y_train, y_val = train_test_split(X_2_augmented, y, test_size=0.2, shuffle=False)
        y_pred_2 = self.model_2.predict(X_2_val)
        
        return y_pred_2, y_val
    
# Cross Validation
def timeseries_cv(df):
    """
    Time Series 5-fold Cross Validation
    Train Set Size = 15years (15*24*365 hours)
    Test Set Size = 1years (1*24*365 hours)
    Gap = 1day (24 hours)
    booster = gbtree
    n_estimators = 800 (more of it may result in overfit!)
    """
    # one-year for test 
    ts_split = TimeSeriesSplit(n_splits=5, test_size=24*365, gap=24)
    fig, axis = plt.subplots(5,1, figsize=(15,15), sharex = True) 
    fold=0
    predicts = list()
    scores = list()

    for train_idx, val_idx in ts_split.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        
        train["PJME_MW"].plot(ax=axis[fold],
                              label="Training Set",)
        test["PJME_MW"].plot(ax=axis[fold],
                              label="Test Set",
                              title=f"Data Train/Test Split fold {fold}")
        axis[fold].axvline(train.index.max(), color = "black", ls="--")
        fold+=1
        
        X_train, y_train = generate_tsFeatures(train, label="PJME_MW")
        X_test, y_test = generate_tsFeatures(test, label="PJME_MW")
        
        XGBR = xgb.XGBRegressor(n_estimators = 800,
                                base_score = 0.5,
                                booster = "gbtree",
                                max_depth = 3,
                                objective="reg:linear",
                                early_stopping_rounds = 50,
                                learning_rate = 0.01)
        XGBR.fit(X_train, y_train,
                eval_set = [(X_train,y_train), (X_test, y_test)],
                verbose = 100)
        
        y_pred = XGBR.predict(X_test)
        predicts.append(y_pred)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(mse)
        
    plt.show()
    print(f"Score across folds {np.mean(scores):0.4f}")
    print(f"Fold Scores: {scores}")
    
def cluster(df):
    """
    k-Means Applied for Clustering, try k-Medeoids too!!
    """
    cluster_mapping = {0: "normal", 1: "low", 
                       2: "high", 3:"extreme"}
    col = np.array(df["PJME_MW"]).reshape(-1, 1)
   
    kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++', n_init="auto").fit(col)
    
    df['cluster'] = kmeans.labels_
    df['cluster'] = df['cluster'].map(cluster_mapping)
    label_counts = df['cluster'].value_counts()
    print("Cluster Label Counts:")
    print(label_counts,"\n")
    print("cluster_centers")
    print(kmeans.cluster_centers_)
    return df

def generate_tsFeatures(df,label=None):
    """
    Generates Time Series Features
    """
    # Dictionary mapping months to seasons
    seasons_mapping = {
        1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 
        6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall", 12: "winter"}
    df = df.copy()
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["dayoftheweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week # quite strong pattern on weekly
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df['season'] = df['month'].map(seasons_mapping)
    df['is_holiday'] = df.index.map(is_holiday)
    df =df.drop(["date"],axis=1)
    df = pd.get_dummies(df, columns = ["season","is_holiday"])
    
    mask = ~df.columns.isin(['PJME_MW'])
    X = df.loc[:, mask]
    
    if label:
        y = df[label]
        return X, y
    return X

def generate_lagFeatures(df):
    
    """
    Generates Lag Features for Several Time Intervals
    """    
    df = df.sort_index()
    target_map = df["PJME_MW"].to_dict()
    
    df["lag0"] = (df.index - pd.Timedelta("7 days")).map(target_map)
    df["lag1"] = (df.index - pd.Timedelta("30 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag4"] = (df.index - pd.Timedelta("1092 days")).map(target_map)
    return df

def is_holiday(date):
    """
    Checks if Date is a US Holiday (Data set is scrapped from US)
    """
    return date in us_holidays

def mape(y, y_hat):
    """
    Mean Absolute Percentage Error (MAPE)
    """
    y, y_hat = np.array(y), np.array(y_hat)
    return np.mean(np.abs((y - y_hat) / y)) *100

def model_specification(params):
    XGBR = xgb.XGBRegressor(
        n_estimators = int(params['n_estimators']),
        max_depth = int(params['max_depth']),
        gamma = params['gamma'],
        reg_alpha = int(params['reg_alpha']),
        min_child_weight = int(params['min_child_weight']),
        colsample_bytree = params['colsample_bytree'],
        reg_lambda = params['reg_lambda'])
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    XGBR.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10, verbose=False)

    pred = XGBR.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("SCORE:", rmse)
    return {'loss': rmse, 'status': STATUS_OK}

def hyperparameter_tuning():
    trials = Trials()
    best_hyperparams = fmin(fn=model_specification,
                            space=params,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)

params = {
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'seed': 0
        }

df = pd.read_csv("PJME_hourly.csv", 
                   index_col=[0], 
                   parse_dates=[0])

df.plot(figsize=(15,5),
        style=".",
        title = "PJME in MW (Raw)")
plt.axhline(20000, color="red", ls="--")

df.query("PJME_MW < 20_000").plot(figsize=(15,5), 
                                  style=".", 
                                  title="Outliers" )
# eleminate obvious Outliers
df = df.query("PJME_MW > 19_000").copy()

naked_df = df.copy()

df.plot(figsize=(15,5),
        style=".",
        title = "PJME in MW (Outliers Removed)")
plt.show()

# call cross validation function
#timeseries_cv(df) 

us_holidays = holidays.UnitedStates()
df = generate_lagFeatures(df)
df = cluster(df)
df = pd.get_dummies(df, columns = ["cluster"])

X, y = generate_tsFeatures(df, label="PJME_MW")
merged_df = X.merge(y, left_index=True, right_index=True)
#print(merged_df)

corr = merged_df.corr(method='pearson')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix of All Data Set")
# plt.show()

def stack_operator(x1,x2,y):
    model_SH = StackedHybrid(
                            model_1 = LinearRegression(fit_intercept = False), # -- Weak Learner
                            model_2 = xgb.XGBRegressor(n_estimators = 800,     # -- Strong Learner
                            early_stopping_rounds = 50,
                            learning_rate = 0.01)
                            )
    model_SH.fit(x1, x2, y)   
    pred, actual = model_SH.predict(x1, x2)
    actual = pd.DataFrame(actual)
    pred = pd.Series(pred)
    actual['predictions'] = pred.values 
    print(" - - - - -  Prediction/Actual Dataframe  - - - - - ")
    return actual

x1 = generate_tsFeatures(naked_df) # x1 includes only time series features
x2 = X                             # x2 includes all features

# USE HybridStack Class
final_df = stack_operator(x1,x2,y)

# RMSE 
rsme_score = np.sqrt(mean_squared_error(final_df["PJME_MW"], final_df["predictions"]))
print(f"RMSE Score on Test set: {rsme_score:0.2f}")

# MAPE 
mape_score = mape(final_df["PJME_MW"], final_df["predictions"])
print(f"Mean Absolute Percantage Error: %{mape_score:0.2f}")

final_df["Gap Percentage(%)"] = (((final_df["PJME_MW"] - final_df["predictions"]) / final_df["PJME_MW"])) * 100 
print(final_df)

# Only XGBoost Model
train = df.loc[df.index < "2017-01-01"]
test = df.loc[df.index >= "2017-01-01"]

fig, ax = plt.subplots(figsize=(15,5))
plt.plot(train.index, train["PJME_MW"])
plt.plot(test.index, test["PJME_MW"])
ax.axvline(train.index.max(), color = "black", ls="--")
ax.legend(["Training Set", "Test Set"])
plt.title("Data set Train/Test Split")
plt.show()

X_train, y_train = generate_tsFeatures(train, label="PJME_MW")
X_test, y_test = generate_tsFeatures(test, label="PJME_MW")
# # features_and_target = pd.concat([X,y], axis =1)

# call hyperparameter tuning function
# hyperparameter_tuning() #

xgb = xgb.XGBRegressor(n_estimators = 800,
                        booster = "gbtree",
                        early_stopping_rounds = 50,
                        learning_rate = 0.01)

xgb.fit(X_train, y_train,
        eval_set = [(X_train,y_train), (X_test, y_test)],
        eval_metric="rmse",
        verbose = 10)

feature_importance = pd.DataFrame(data = xgb.feature_importances_,
                                  index = xgb.feature_names_in_,
                                  columns = ["importance"])

feature_importance.sort_values("importance").plot(kind="barh",title="Feature Importance")

test["prediction"] = xgb.predict(X_test)
test.index = pd.to_datetime(test.index)

df = df.merge(test[["prediction"]], how="left", left_index=True, right_index=True)

ax = df[["PJME_MW"]].plot(figsize = (15,5))
df["prediction"].plot(ax=ax)
plt.legend(["Past Data", "Predicted Data"])
ax.set_title("Energy Predictions in MW")
plt.show()

# RMSE 
rsme_score = np.sqrt(mean_squared_error(test["PJME_MW"], test["prediction"]))
print(f"RMSE Score on Test set: {rsme_score:0.2f}")

# MAPE 
mape_score = mape(test["PJME_MW"], test["prediction"])
print(f"Mean Absolute Percantage Error: %{mape_score:0.2f}")

test["error"] = np.abs(test["PJME_MW"] - test["prediction"])
test["date"] = test.index.date

worst_predicted = test.groupby("date")["error"].mean().sort_values(ascending = False).head(10)
best_predicted = test.groupby("date")["error"].mean().sort_values(ascending = True).head(10)

print(f"\nTop 10 Worst Prediction by date: {worst_predicted}\n")
print(f"Top 10 Best Prediction by date: {best_predicted}")