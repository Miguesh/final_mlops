import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import pandas as pd


X_test = pd.read_csv("data/X_test_processed.csv")
X_train = pd.read_csv("data/X_train_processed.csv")
y_test = pd.read_csv("data/y_test.csv")
y_train = pd.read_csv("data/y_train.csv")

X = pd.concat([X_test,X_train])
y = pd.concat([y_test,y_train])
# =====================================================
#   OBJECTIVE RANDOM FOREST
# =====================================================
def objective_rf(trial):
    with mlflow.start_run(nested=True):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_params(trial.params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        return r2


# =====================================================
#   OBJECTIVE XGBOOST
# =====================================================
def objective_xgb(trial):
    with mlflow.start_run(nested=True):
        eta = trial.suggest_float("eta", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        n_estimators = trial.suggest_int("n_estimators", 100, 400)

        model = XGBRegressor(
            eta=eta,
            max_depth=max_depth,
            n_estimators=n_estimators
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred) 
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_params(trial.params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        return r2


# =====================================================
#   OBJECTIVE SVR
# =====================================================
def objective_svr(trial):
    with mlflow.start_run(nested=True):
        C = trial.suggest_float("C", 0.1, 10)
        gamma = trial.suggest_float("gamma", 0.0001, 1)

        model = SVR(C=C, gamma=gamma)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        mlflow.log_param("model", "SVR")
        mlflow.log_params(trial.params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        return r2


# =====================================================
#   EJECUTAR LOS 3 ESTUDIOS
# =====================================================
mlflow.set_experiment("Final MLOPS 3")

with mlflow.start_run(run_name="Entrenamiento Completo Preprocesado"):

    # RF
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=20)

    # XGB
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=20)

    # SVR
    study_svr = optuna.create_study(direction="minimize")
    study_svr.optimize(objective_svr, n_trials=20)

    # ===============================
    #  GUARDAR LOS MEJORES MODELOS
    # ===============================

    best_rf = RandomForestRegressor(**study_rf.best_params)
    best_rf.fit(X, y)
    mlflow.sklearn.log_model(best_rf, "best_random_forest")

    best_xgb = XGBRegressor(**study_xgb.best_params)
    best_xgb.fit(X, y)
    mlflow.sklearn.log_model(best_xgb, "best_xgboost")

    best_svr = SVR(**study_svr.best_params)
    best_svr.fit(X, y)
    mlflow.sklearn.log_model(best_svr, "best_svr")
