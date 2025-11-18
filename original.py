##best submission

import os
import pandas as pd
import numpy as np
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



class ValueCountMapper:
    def __init__(self, normalize=True, suffix="_freq"):
        self.normalize = normalize
        self.suffix = suffix
        self.count_maps_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.columns_ = [str(c) for c in X_df.columns.tolist()]
        self.count_maps_ = {} 
        
        for idx, orig_col in enumerate(X_df.columns):
            col_str = self.columns_[idx]
            
            vc = X_df[orig_col].value_counts(
                normalize=self.normalize, 
                dropna=False
            )
            self.count_maps_[col_str] = vc.to_dict()
        
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns_)
        out = pd.DataFrame(index=X_df.index)
        
        for col in self.columns_:
            mapping = self.count_maps_.get(col, {})
            
            new_col_name = str(col) + self.suffix
            out[new_col_name] = X_df[col].map(mapping).fillna(0.0).astype(float)
            
        return out.values

    def get_feature_names_out(self, input_features=None):
        cols = input_features if input_features is not None else self.columns_
        return [str(c) + self.suffix for c in cols]


def make_ohe(sparse_output=False):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not sparse_output)


train_path = r"train.csv"
test_path  = r"test.csv"

ID_COL = "id"
TARGET = "Lap_Time_Seconds"

cat_onehot = ['Formula_category_x', 'Formula_Track_Condition', 'Tire_Compound',
              'Penalty', 'Session', 'weather', 'track']
cat_freq = ['Formula_shortname', 'circuit_name']

numeric_cols_all = ['id', 'Unique ID', 'Rider_ID', 'Len_Circuit_inkm', 'Laps',
                    'Start_Position', 'Formula_Avg_Speed_kmh', 'Humidity_%',
                    'Champ_Points', 'Champ_Position', 'race_year', 'seq', 'position',
                    'points', 'Corners_in_Lap', 'Tire_Degradation_Factor_per_Lap',
                    'Pit_Stop_Duration_Seconds', 'Ambient_Temperature_Celsius',
                    'Track_Temperature_Celsius', 'air', 'ground', 'starts',
                    'finishes', 'with_points', 'podiums', 'wins', 'Lap_Time_Seconds']


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rf_params = dict(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)

et_params = dict(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, bootstrap=False
)

USE_SPARSE_OHE = False

print("Loading train/test CSVs...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

if ID_COL not in train_df.columns or TARGET not in train_df.columns:
    raise ValueError("train.csv must contain 'id' and 'Lap_Time_Seconds' columns.")
if ID_COL not in test_df.columns:
    raise ValueError("test.csv must contain 'id' column.")

cat_onehot = [c for c in cat_onehot if c in train_df.columns]
cat_freq = [c for c in cat_freq if c in train_df.columns]
numeric_features = [c for c in numeric_cols_all if c in train_df.columns and c not in (ID_COL, TARGET)]

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"One-hot categorical ({len(cat_onehot)}): {cat_onehot}")
print(f"Freq categorical ({len(cat_freq)}): {cat_freq}")


if 'Penalty' in train_df.columns:
    train_df['Penalty'] = train_df['Penalty'].fillna('NoPenalty')
if 'Penalty' in test_df.columns:
    test_df['Penalty'] = test_df['Penalty'].fillna('NoPenalty')

X_full = train_df.drop(columns=[TARGET])
y_full = train_df[TARGET].values


def build_preprocessor(onehot_cols, freq_cols, numeric_cols, use_sparse_ohe=USE_SPARSE_OHE):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    onehot_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", make_ohe(sparse_output=use_sparse_ohe))
    ])
    freq_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("mapper", ValueCountMapper(normalize=True, suffix="_freq"))
    ])

    transformers = []
    if onehot_cols:
        transformers.append(("onehot", onehot_transformer, onehot_cols))
    if freq_cols:
        transformers.append(("freq_map", freq_transformer, freq_cols))
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    return ColumnTransformer(
        transformers=transformers, 
        remainder="drop", 
        verbose_feature_names_out=False 
    )


models = [
    ("rf", RandomForestRegressor(**rf_params)),
    ("et", ExtraTreesRegressor(**et_params))
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

        pre = build_preprocessor(cat_onehot, cat_freq, numeric_features)
        pre.fit(X_tr)

        X_tr_t = pre.transform(X_tr)
        X_va_t = pre.transform(X_va)
        X_test_t = pre.transform(test_df)

        feat_names = pre.get_feature_names_out()

        X_tr_df = pd.DataFrame(X_tr_t, columns=feat_names)
        X_va_df = pd.DataFrame(X_va_t, columns=feat_names)
        X_test_df = pd.DataFrame(X_test_t, columns=feat_names)

        model = RandomForestRegressor(**rf_params) if model_name == "rf" else ExtraTreesRegressor(**et_params)
        model.fit(X_tr_df, y_tr)

        pred_val = model.predict(X_va_df)
        oof_preds[val_idx] = pred_val

        rmse = np.sqrt(mean_squared_error(y_va, pred_val))
        fold_rmse.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.4f}")

        test_preds_cv += model.predict(X_test_df) / n_splits

    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)
    overall_oof = np.sqrt(mean_squared_error(y_full, oof_preds))

    print(f"\n{model_name.upper()} CV mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"{model_name.upper()} OOF RMSE: {overall_oof:.4f}")

    print(f"\nTraining final {model_name} model on full data...")
    pre_full = build_preprocessor(cat_onehot, cat_freq, numeric_features)
    pre_full.fit(X_full)
    X_full_t = pre_full.transform(X_full)
    X_test_t = pre_full.transform(test_df)
    
    feat_names_full = pre_full.get_feature_names_out()
    
    X_full_df = pd.DataFrame(X_full_t, columns=feat_names_full)
    X_test_df = pd.DataFrame(X_test_t, columns=feat_names_full)

    final_model = RandomForestRegressor(**rf_params) if model_name == "rf" else ExtraTreesRegressor(**et_params)
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

print("All done.")