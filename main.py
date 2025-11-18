##best submission

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")


from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor



train_path = r"final_cleaned_train.csv"
test_path  = r"final_cleaned_test.csv"

ID_COL = "id"
TARGET = "Lap_Time_Seconds"

numeric_cols_all = ['circuit_name_estLap_std', 'points_finish_rate', ',WithPoints_Ratio', 'position_points_efficiency', 'with_points', 'points', 'Corners_in_Lap', 'temp_ratio_air_ground', 'corners_per_lap', 'temp_diff', 'Consistency_Score', 'position', 'finishes', 'finish_rate', 'Lap_Time_Seconds']


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rf_params = dict(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)

et_params = dict(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, bootstrap=False
)

xgb_params = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1
)

USE_SPARSE_OHE = False

print("Loading train/test CSVs...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

if ID_COL not in train_df.columns or TARGET not in train_df.columns:
    raise ValueError("train.csv must contain 'id' and 'Lap_Time_Seconds' columns.")
if ID_COL not in test_df.columns:
    raise ValueError("test.csv must contain 'id' column.")

numeric_features = [c for c in numeric_cols_all if c in train_df.columns and c not in (ID_COL, TARGET)]

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")

X_full = train_df.drop(columns=[TARGET])
y_full = train_df[TARGET].values


def build_preprocessor(numeric_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    return ColumnTransformer(
        transformers=transformers, 
        remainder="drop", 
        verbose_feature_names_out=False 
    )


models = [
    ("rf", RandomForestRegressor(**rf_params)),
    ("et", ExtraTreesRegressor(**et_params)),
    ("lr", LinearRegression()),
    ("xgb", XGBRegressor(**xgb_params)),
]


for model_name, model_obj in models:
    print("\n" + "=" * 40)
    print(f"Running model: {model_name}")
    print("=" * 40)

    test_preds_cv = np.zeros(len(test_df))
    oof_preds = np.zeros(len(train_df))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full), 1):
        print(f"\n-- Fold {fold}/{n_splits} --")
        X_tr, X_va = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]

        pre = build_preprocessor(numeric_features)
        pre.fit(X_tr)

        X_tr_t = pre.transform(X_tr)
        X_va_t = pre.transform(X_va)
        X_test_t = pre.transform(test_df)

        feat_names = pre.get_feature_names_out()

        X_tr_df = pd.DataFrame(X_tr_t, columns=feat_names)
        X_va_df = pd.DataFrame(X_va_t, columns=feat_names)
        X_test_df = pd.DataFrame(X_test_t, columns=feat_names)

        if model_name == 'rf':
            model = RandomForestRegressor(**rf_params)
        elif model_name == 'et':
            model = ExtraTreesRegressor(**et_params)
        elif model_name == 'lr':
            model = LinearRegression()
        else:
            model = XGBRegressor(**xgb_params)

        model.fit(X_tr_df, y_tr)

        pred_val = model.predict(X_va_df)
        oof_preds[val_idx] = pred_val

        rmse = np.sqrt(mean_squared_error(y_va, pred_val))
        fold_rmse.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.4f}")

        test_preds_cv += model.predict(X_test_df) / n_splits

    plt.figure(figsize=(6,4))
    plt.plot(range(1, n_splits+1), fold_rmse, marker='o')
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title(f"{model_name.upper()} RMSE per Fold")
    plt.grid(True)
    plt.savefig(f"{model_name}_rmse_per_fold.png")
    plt.close()

    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)
    overall_oof = np.sqrt(mean_squared_error(y_full, oof_preds))

    print(f"\n{model_name.upper()} CV mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"{model_name.upper()} OOF RMSE: {overall_oof:.4f}")

    print(f"\nTraining final {model_name} model on full data...")
    pre_full = build_preprocessor(numeric_features)
    pre_full.fit(X_full)
    X_full_t = pre_full.transform(X_full)
    X_test_t = pre_full.transform(test_df)
    
    feat_names_full = pre_full.get_feature_names_out()
    
    X_full_df = pd.DataFrame(X_full_t, columns=feat_names_full)
    X_test_df = pd.DataFrame(X_test_t, columns=feat_names_full)

    if model_name == 'rf':
        final_model = RandomForestRegressor(**rf_params)
    elif model_name == 'et':
        final_model = ExtraTreesRegressor(**et_params)
    elif model_name == 'lr':
        final_model = LinearRegression()
    else:
        final_model = XGBRegressor(**xgb_params)
    final_model.fit(X_full_df, y_full)

    joblib.dump(final_model, f"{model_name}_final_model.joblib")
    joblib.dump(pre_full, f"{model_name}_preprocessor.joblib")
    print(f"Saved {model_name}_final_model.joblib and {model_name}_preprocessor.joblib")

    test_pred_final = final_model.predict(X_test_df)

    out_cv_avg = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_preds_cv})
    out_cv_avg.to_csv(f"output_cv_avg_{model_name}.csv", index=False)

    out_final = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_pred_final})
    if model_name == "et":
        out_final.to_csv("outputET.csv", index=False)
    else:
        out_final.to_csv(f"output_{model_name}.csv", index=False)

    print(f"Finished model: {model_name}\n")

print("All done.")