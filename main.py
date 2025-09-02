"""
EDA Streamlit Pro — Polars + Streamlit + AutoML (完全版)

Features:
- Polars-based fast CSV handling (lazy scan + sampling)
- Interactive Plotly visualizations with polished styling
- Automatic type inference and recommended statistical tests
- t-tests, Mann-Whitney U, ANOVA, chi-square
- PCA and KMeans clustering + visualizations
- Feature Engineering (OneHot, Text features, DateTime, Interactions)
- Outlier detection (IQR + Z-score)
- Intuitive file-merge UI with preview and download
- PDF & Markdown report generation (uses kaleido + fpdf)
- **NEW: AutoML with LightGBM, XGBoost, RandomForest**
- **NEW: Optuna hyperparameter optimization**
- **NEW: SHAP interpretability**
- **NEW: Model comparison dashboard**
- Caching for performance and responsive behavior for large files

Run:
pip install streamlit polars pandas numpy scipy scikit-learn plotly fpdf kaleido lightgbm xgboost optuna shap joblib
streamlit run eda_streamlit_pro_automl.py

Notes:
- For very large files the app uses smart sampling for plotting and exploratory summaries.
- Feature engineering allows analysis of any data type (text, categorical, numeric)
- AutoML supports both regression and classification with automatic task detection
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
import base64
import tempfile
import os
import time
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For PDF export
from fpdf import FPDF

# === AutoML extension ===
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Streamlit page config
st.set_page_config(page_title="EDA Studio Pro + AutoML", layout="wide", initial_sidebar_state='expanded')

# ---------------------- Styles & utilities ----------------------
st.markdown("""
<style>
/* Glassy container */
[data-testid='stAppViewContainer'] { background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%); }
.stButton>button { background: linear-gradient(90deg,#8b5cf6,#ec4899); color: white; border: none; }
.metric-container { 
    background: rgba(139, 92, 246, 0.1); 
    padding: 1rem; 
    border-radius: 10px; 
    border: 1px solid rgba(139, 92, 246, 0.3);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def read_csv_polars(file_buf, use_lazy=True, sample_rows=20000):
    file_buf.seek(0)
    try:
        if use_lazy:
            lf = pl.scan_csv(file_buf)
            sample = lf.limit(sample_rows).collect()
            pl_df = sample
            total_rows = None
            try:
                total_rows = lf.collect().height
                pl_df = lf.collect()
            except Exception:
                total_rows = None
        else:
            pl_df = pl.read_csv(file_buf)
            total_rows = pl_df.height
    except Exception as e:
        # fallback to pandas
        file_buf.seek(0)
        pdf = pd.read_csv(file_buf)
        pl_df = pl.from_pandas(pdf)
        total_rows = pl_df.height

    # create sampled pandas for plotting
    try:
        n = pl_df.height
        if n > 20000:
            sampled = pl_df.sample(n=20000, with_replacement=False)
        else:
            sampled = pl_df
        sampled_pd = sampled.to_pandas()
    except Exception:
        sampled_pd = pl_df.head(1000).to_pandas()

    return pl_df, sampled_pd

@st.cache_data
def infer_column_types(sampled_pd):
    types = {}
    for c in sampled_pd.columns:
        s = sampled_pd[c]
        if pd.api.types.is_numeric_dtype(s):
            types[c] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[c] = 'datetime'
        else:
            # treat string-like but with low unique count as categorical
            nunique = s.nunique(dropna=True)
            if nunique <= min(50, max(10, int(len(s)*0.05))):
                types[c] = 'categorical'
            else:
                types[c] = 'text'
    return types

@st.cache_data
def compute_numeric_stats(arr):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    return {
        'count': int(a.size),
        'mean': float(np.mean(a)),
        'std': float(np.std(a, ddof=0)),
        'median': float(np.median(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'q1': float(np.quantile(a,0.25)),
        'q3': float(np.quantile(a,0.75))
    }

@st.cache_data
def compute_basic_stats(sampled_pd, types):
    stats_dict = {}
    for c,t in types.items():
        if t == 'numeric':
            stats_dict[c] = compute_numeric_stats(sampled_pd[c].dropna())
        elif t in ('categorical','text'):
            vc = sampled_pd[c].value_counts().head(100)
            stats_dict[c] = { 
                'count': int(sampled_pd[c].notna().sum()), 
                'unique': int(sampled_pd[c].nunique(dropna=True)), 
                'top': vc.index[0] if vc.shape[0]>0 else None, 
                'value_counts': vc
            }
        else:
            stats_dict[c] = { 'info': 'datetime or other' }
    return stats_dict

# OneHot Encoding and Feature Engineering utilities
@st.cache_data
def create_onehot_features(sampled_pd, categorical_cols, max_categories=10):
    """Create OneHot encoded features from categorical columns"""
    encoded_df = sampled_pd.copy()
    encoding_info = {}
    
    for col in categorical_cols:
        # Get top categories to avoid too many dummy variables
        top_categories = sampled_pd[col].value_counts().head(max_categories).index.tolist()
        
        # Create dummy variables for top categories
        dummies = pd.get_dummies(sampled_pd[col], prefix=f'{col}', prefix_sep='_')
        
        # Only keep top categories
        dummy_cols = [f'{col}_{cat}' for cat in top_categories if f'{col}_{cat}' in dummies.columns]
        selected_dummies = dummies[dummy_cols]
        
        # Merge with main dataframe
        encoded_df = pd.concat([encoded_df, selected_dummies], axis=1)
        
        encoding_info[col] = {
            'original_unique': sampled_pd[col].nunique(),
            'encoded_cols': dummy_cols,
            'top_categories': top_categories
        }
    
    return encoded_df, encoding_info

@st.cache_data
def create_numeric_from_text(sampled_pd, text_cols):
    """Create numeric features from text columns"""
    numeric_features = {}
    
    for col in text_cols:
        # Length of text
        numeric_features[f'{col}_length'] = sampled_pd[col].astype(str).str.len()
        
        # Word count
        numeric_features[f'{col}_word_count'] = sampled_pd[col].astype(str).str.split().str.len()
        
        # Number of unique characters
        numeric_features[f'{col}_unique_chars'] = sampled_pd[col].astype(str).apply(lambda x: len(set(x)))
        
        # Number of digits
        numeric_features[f'{col}_digit_count'] = sampled_pd[col].astype(str).str.count(r'\d')
        
        # Number of uppercase letters
        numeric_features[f'{col}_upper_count'] = sampled_pd[col].astype(str).str.count(r'[A-Z]')
        
        # Contains specific patterns (email, URL, etc.)
        numeric_features[f'{col}_has_email'] = sampled_pd[col].astype(str).str.contains(r'@', case=False, na=False).astype(int)
        numeric_features[f'{col}_has_url'] = sampled_pd[col].astype(str).str.contains(r'http', case=False, na=False).astype(int)
    
    return pd.DataFrame(numeric_features)

@st.cache_data
def create_datetime_features(sampled_pd, datetime_cols):
    """Create numeric features from datetime columns"""
    datetime_features = {}
    
    for col in datetime_cols:
        # Convert to datetime if not already
        try:
            dt_series = pd.to_datetime(sampled_pd[col], errors='coerce')
        except:
            continue
            
        # Extract various datetime components
        datetime_features[f'{col}_year'] = dt_series.dt.year
        datetime_features[f'{col}_month'] = dt_series.dt.month
        datetime_features[f'{col}_day'] = dt_series.dt.day
        datetime_features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
        datetime_features[f'{col}_hour'] = dt_series.dt.hour
        datetime_features[f'{col}_quarter'] = dt_series.dt.quarter
        datetime_features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
        
        # Time-based calculations
        reference_date = dt_series.min()
        datetime_features[f'{col}_days_since_start'] = (dt_series - reference_date).dt.days
    
    return pd.DataFrame(datetime_features)

@st.cache_data
def create_interaction_features(sampled_pd, numeric_cols, max_interactions=10):
    """Create interaction features between numeric columns"""
    interaction_features = {}
    
    # Limit to prevent too many features
    limited_cols = numeric_cols[:max_interactions]
    
    for i, col1 in enumerate(limited_cols):
        for col2 in limited_cols[i+1:]:
            # Multiplication
            interaction_features[f'{col1}_x_{col2}'] = sampled_pd[col1] * sampled_pd[col2]
            
            # Division (with protection against division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                division = sampled_pd[col1] / (sampled_pd[col2] + 1e-8)
                division = np.where(np.isfinite(division), division, 0)
                interaction_features[f'{col1}_div_{col2}'] = division
            
            # Addition
            interaction_features[f'{col1}_plus_{col2}'] = sampled_pd[col1] + sampled_pd[col2]
            
            # Subtraction
            interaction_features[f'{col1}_minus_{col2}'] = sampled_pd[col1] - sampled_pd[col2]
    
    return pd.DataFrame(interaction_features)

def create_binned_features(sampled_pd, numeric_cols, n_bins=5):
    """Create categorical features by binning numeric columns"""
    binned_features = {}
    
    for col in numeric_cols:
        try:
            # Use quantile-based binning
            binned_features[f'{col}_binned'] = pd.qcut(
                sampled_pd[col], 
                q=n_bins, 
                labels=[f'{col}_bin_{i}' for i in range(n_bins)],
                duplicates='drop'
            )
        except:
            # Fallback to equal-width binning
            binned_features[f'{col}_binned'] = pd.cut(
                sampled_pd[col], 
                bins=n_bins, 
                labels=[f'{col}_bin_{i}' for i in range(n_bins)]
            )
    
    return pd.DataFrame(binned_features)

# Suggest appropriate tests based on types
def suggest_tests_for_columns(types, stats_dict):
    suggestions = []
    # find binary categorical vs numeric combos
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    cat_cols = [c for c,t in types.items() if t=='categorical']

    for n in numeric_cols:
        for g in cat_cols:
            if g in stats_dict and 'unique' in stats_dict[g] and stats_dict[g]['unique'] == 2:
                suggestions.append({'test': 't-test (independent)', 'numeric': n, 'group': g, 'reason': 'binary group'})
            elif g in stats_dict and 'unique' in stats_dict[g]:
                suggestions.append({'test': 'ANOVA (1-way)', 'numeric': n, 'group': g, 'reason': 'categorical with >2 groups'})

    # categorical vs categorical
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i+1,len(cat_cols)):
                suggestions.append({'test': 'chi-square', 'cat1': cat_cols[i], 'cat2': cat_cols[j]})
    return suggestions

# PCA & clustering utilities
@st.cache_data
def run_pca(sampled_pd, numeric_cols, n_components=3):
    X = sampled_pd[numeric_cols].dropna()
    if X.shape[0] == 0:
        return None
    pca = PCA(n_components=min(n_components, X.shape[1]))
    comps = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_
    comp_df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(comps.shape[1])])
    return comp_df, variance

@st.cache_data
def run_kmeans(sampled_pd, numeric_cols, n_clusters=3):
    X = sampled_pd[numeric_cols].dropna()
    if X.shape[0] == 0:
        return None
    k = KMeans(n_clusters=n_clusters, random_state=42)
    labels = k.fit_predict(X)
    return labels

# Feature importance (light)
@st.cache_data
def compute_feature_importance(sampled_pd, target_col, numeric_cols):
    # train a small RF to get importances (classification)
    df = sampled_pd[numeric_cols + [target_col]].dropna()
    if df.shape[0] < 20:
        return None
    X = df[numeric_cols]
    y = df[target_col]
    # if numeric target, bin to classify
    if pd.api.types.is_numeric_dtype(y):
        y = pd.qcut(y, q=3, labels=False, duplicates='drop')
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
        return importances
    except Exception:
        return None

# === AutoML extension starts here ===

# AutoML preprocessing utilities
def preprocess_for_automl(df, target_col, task_type='auto'):
    """Comprehensive preprocessing for AutoML"""
    df_processed = df.copy()
    
    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Handle missing values in target
    mask = ~y.isnull()
    X = X.loc[mask]
    y = y.loc[mask]
    
    # Determine task type automatically if not specified
    if task_type == 'auto':
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique() / len(y)
            if unique_ratio < 0.05 or y.nunique() <= 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
        else:
            task_type = 'classification'
    
    # Encode target for classification
    target_encoder = None
    if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Preprocessing pipeline
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessing_info = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'task_type': task_type,
        'target_encoder': target_encoder
    }
    
    # Handle numeric columns
    if numeric_cols:
        # Fill numeric missing values
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        preprocessing_info['numeric_imputer'] = numeric_imputer
        
        # Scale numeric features
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        preprocessing_info['scaler'] = scaler
    
    # Handle categorical columns
    if categorical_cols:
        # Fill categorical missing values
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # One-hot encode categorical features (limit to top categories)
        for col in categorical_cols:
            # Limit to top 10 categories to prevent explosion
            top_categories = X[col].value_counts().head(10).index
            X[col] = X[col].apply(lambda x: x if x in top_categories else 'OTHER')
        
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(X[categorical_cols])
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_feature_names, index=X.index)
        
        # Replace categorical columns with encoded ones
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, encoded_df], axis=1)
        
        preprocessing_info['categorical_imputer'] = categorical_imputer
        preprocessing_info['encoder'] = encoder
    
    return X, y, preprocessing_info

def get_cv_strategy(cv_type, n_splits, y, task_type):
    """Get cross-validation strategy"""
    if cv_type == 'StratifiedKFold' and task_type == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_type == 'TimeSeriesSplit':
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

def create_models():
    """Create model instances"""
    models = {
        'LightGBM': {
            'model_class': lgb.LGBMRegressor,
            'classifier_class': lgb.LGBMClassifier,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        },
        'XGBoost': {
            'model_class': xgb.XGBRegressor,
            'classifier_class': xgb.XGBClassifier,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        },
        'RandomForest': {
            'model_class': RandomForestRegressor,
            'classifier_class': RandomForestClassifier,
            'param_space': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        }
    }
    return models

def create_optuna_objective(model_class, X_train, y_train, cv_strategy, task_type, param_space):
    """Create Optuna objective function"""
    def objective(trial):
        # Sample hyperparameters
        params = {}
        for param, values in param_space.items():
            if isinstance(values, list):
                if all(isinstance(v, int) for v in values):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif all(isinstance(v, float) for v in values):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
        
        # Add common parameters
        params['random_state'] = 42
        if 'LGB' in str(model_class):
            params['verbose'] = -1
            params['force_col_wise'] = True
        elif 'XGB' in str(model_class):
            params['verbosity'] = 0
        elif 'RandomForest' in str(model_class):
            params['n_jobs'] = -1
        
        # Create model
        model = model_class(**params)
        
        # Cross-validation
        scoring = 'roc_auc' if task_type == 'classification' else 'neg_root_mean_squared_error'
        try:
            cv_results = cross_validate(
                model, X_train, y_train, 
                cv=cv_strategy, 
                scoring=scoring, 
                n_jobs=1,  # Avoid nested parallelism
                error_score='raise'
            )
            return np.mean(cv_results['test_score'])
        except Exception as e:
            return float('-inf')  # Return bad score on failure
    
    return objective

def run_automl_optimization(X_train, y_train, model_name, task_type, cv_type, n_splits, n_trials, timeout_minutes):
    """Run Optuna hyperparameter optimization"""
    models = create_models()
    model_info = models[model_name]
    
    # Select appropriate model class
    model_class = model_info['classifier_class'] if task_type == 'classification' else model_info['model_class']
    param_space = model_info['param_space']
    
    # Create CV strategy
    cv_strategy = get_cv_strategy(cv_type, n_splits, y_train, task_type)
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    objective = create_optuna_objective(model_class, X_train, y_train, cv_strategy, task_type, param_space)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_minutes * 60,
        show_progress_bar=False  # Will show custom progress in Streamlit
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params['random_state'] = 42
    
    # Add model-specific parameters
    if 'LGB' in str(model_class):
        best_params['verbose'] = -1
        best_params['force_col_wise'] = True
    elif 'XGB' in str(model_class):
        best_params['verbosity'] = 0
    elif 'RandomForest' in str(model_class):
        best_params['n_jobs'] = -1
    
    # Train final model
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    
    return {
        'model': best_model,
        'best_params': best_params,
        'best_score': study.best_value,
        'study': study
    }

def evaluate_model(model, X_test, y_test, task_type):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    results = {}
    
    if task_type == 'classification':
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['f1'] = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC and PR-AUC for binary classification
        if len(np.unique(y_test)) == 2:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            results['pr_auc'] = average_precision_score(y_test, y_pred_proba)
        
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
    else:  # regression
        results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2'] = r2_score(y_test, y_pred)
    
    results['predictions'] = y_pred
    
    return results

def compute_feature_importance_automl(model, feature_names, importance_type='gain'):
    """Compute feature importance for AutoML models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

def compute_shap_values(model, X_sample, max_samples=100):
    """Compute SHAP values for model interpretation"""
    try:
        # Limit sample size for performance
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=42)
        
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):  # Classification
            explainer = shap.TreeExplainer(model)
        else:  # Regression or other
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class classification case
            if len(shap_values) > 1:
                # Use positive class for binary classification
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        
        # Ensure 2D shape
        if len(shap_values.shape) == 3:
            # If still 3D, take first dimension
            shap_values = shap_values[0]
        
        # Final validation
        if len(shap_values.shape) != 2:
            st.warning(f"SHAP値の形状が予期しません: {shap_values.shape}")
            return None, None, None
            
        if shap_values.shape[0] != len(X_sample) or shap_values.shape[1] != len(X_sample.columns):
            st.warning(f"SHAP値の次元が一致しません: {shap_values.shape} vs expected {(len(X_sample), len(X_sample.columns))}")
            return None, None, None
        
        return shap_values, explainer, X_sample
        
    except Exception as e:
        st.warning(f"SHAP計算でエラーが発生: {str(e)}")
        # Additional debugging info
        if 'shap_values' in locals():
            st.warning(f"SHAP値の形状: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
        return None, None, None

# Report generation: create pictures from plotly via to_image (requires kaleido)
def plotly_to_png(fig):
    try:
        img_bytes = fig.to_image(format='png', engine='kaleido')
        return img_bytes
    except Exception as e:
        return None

def create_pdf_report(title, description, images_with_captions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, description)
    for img_bytes, caption in images_with_captions:
        if img_bytes is None:
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(img_bytes)
        tmp.close()
        pdf.add_page()
        pdf.image(tmp.name, x=10, y=20, w=190)
        pdf.ln(95)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, caption)
        os.unlink(tmp.name)
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ---------------------- App layout ----------------------
st.title('EDA Studio Pro + AutoML — Polars + Streamlit')
st.caption('Fast, polished EDA and statistical analysis with AutoML capabilities')

left_col, right_col = st.columns([3,1])

# Sidebar controls
with right_col:
    st.header('設定')
    use_lazy = st.checkbox('Lazy読み込み (Polars scan_csv)', value=True)
    sample_size = st.number_input('サンプル行数 (プロット)', min_value=2000, max_value=200000, value=20000, step=1000)
    display_rows = st.number_input('プレビュー行数', min_value=20, max_value=1000, value=200)
    
    st.markdown('---')
    st.write('ファイル操作')
    uploaded = st.file_uploader('CSV をアップロード', type=['csv'], accept_multiple_files=True)
    st.write('またはサンプルを使用')
    if st.button('サンプルデータ生成'):
        sample_pdf = pd.DataFrame({
            'age': np.random.randint(22,65,5000),
            'income': (np.random.normal(500000,120000,5000)).astype(int),
            'experience': np.random.randint(0,30,5000),
            'education': np.random.choice(['HS','BSc','MSc','PhD'],5000),
            'group': np.random.choice(['A','B','C'],5000),
            'outcome': np.random.choice([0,1],5000,p=[0.8,0.2])
        })
        buf = StringIO()
        sample_pdf.to_csv(buf, index=False)
        buf.seek(0)
        st.session_state['uploaded_sample'] = buf.getvalue()

    # === AutoML Sidebar Controls ===
    st.markdown('---')
    st.write('🤖 AutoML設定')
    
    # Initialize session state for AutoML if not exists
    if 'automl_target' not in st.session_state:
        st.session_state['automl_target'] = None

# load files
file_store = {}
if uploaded:
    for f in uploaded:
        try:
            pl_df, sampled_pd = read_csv_polars(f, use_lazy=use_lazy, sample_rows=sample_size)
            file_store[f.name] = (pl_df, sampled_pd)
        except Exception as e:
            st.error(f'{f.name} 読み込み失敗: {e}')

if 'uploaded_sample' in st.session_state and not uploaded:
    buf = StringIO(st.session_state['uploaded_sample'])
    pl_df, sampled_pd = read_csv_polars(buf, use_lazy=False, sample_rows=sample_size)
    file_store['sample.csv'] = (pl_df, sampled_pd)

if not file_store:
    st.warning('CSVをアップロードするかサンプルを生成してください')
    st.stop()

# Select file to analyze (main area)
file_names = list(file_store.keys())
selected_file = left_col.selectbox('解析ファイルを選択', options=file_names)
pl_df, sampled_pd = file_store[selected_file]

types = infer_column_types(sampled_pd)
basic_stats = compute_basic_stats(sampled_pd, types)

# Main tabs - Added AutoML tab
tabs = left_col.tabs(['Overview','Visualization','Feature Engineering','Stat Tests','PCA & Clustering','AutoML','Merge & Export','Report'])

# -------- Overview --------
with tabs[0]:
    st.subheader('概要')
    c1,c2,c3 = st.columns(3)
    c1.metric('行数 (サンプル表示)', sampled_pd.shape[0])
    c2.metric('列数', sampled_pd.shape[1])
    numeric_count = sum(1 for v in types.values() if v=='numeric')
    c3.metric('数値列', numeric_count)
    st.write('---')
    st.write('先頭行（サンプル）')
    st.dataframe(sampled_pd.head(display_rows))
    st.write('---')
    st.write('列タイプサマリ')
    st.table(pd.Series(types).rename('inferred_type'))
    st.write('---')

# -------- Visualization --------
with tabs[1]:
    st.subheader('可視化')
    col_choice = st.selectbox('列を選択 (可視化)', options=sampled_pd.columns)
    if col_choice:
        if types[col_choice]=='numeric':
            fig = px.histogram(sampled_pd, x=col_choice, nbins=80, title=f'分布: {col_choice}', marginal='box')
            left_col.plotly_chart(fig, use_container_width=True)
            # box + outliers
            if col_choice in basic_stats and basic_stats[col_choice] is not None:
                q1 = basic_stats[col_choice]['q1']; q3 = basic_stats[col_choice]['q3']; iqr = q3 - q1
                lower = q1 - 1.5*iqr; upper = q3 + 1.5*iqr
                outliers = sampled_pd[(sampled_pd[col_choice] < lower) | (sampled_pd[col_choice] > upper)]
                st.write(f'外れ値 (IQR基準) 推定: {len(outliers)}')
        else:
            vc = sampled_pd[col_choice].value_counts().reset_index()
            vc.columns = [col_choice, 'count']
            fig = px.bar(vc.head(100), x=col_choice, y='count', title=f'頻度: {col_choice}')
            left_col.plotly_chart(fig, use_container_width=True)

    st.write('---')
    # correlation
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    if len(numeric_cols) >= 2:
        corr = sampled_pd[numeric_cols].corr()
        figc = px.imshow(corr, text_auto='.2f', title='相関行列')
        left_col.plotly_chart(figc, use_container_width=True)
        if left_col.button('散布図行列 (上位6列)'):
            figm = px.scatter_matrix(sampled_pd[numeric_cols].sample(n=min(2000, sampled_pd.shape[0])), dimensions=numeric_cols[:6])
            left_col.plotly_chart(figm, use_container_width=True)

# -------- Feature Engineering --------
with tabs[2]:
    st.subheader('特徴量エンジニアリング')
    
    # Get column types
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    categorical_cols = [c for c,t in types.items() if t=='categorical']
    text_cols = [c for c,t in types.items() if t=='text']
    datetime_cols = [c for c,t in types.items() if t=='datetime']
    
    st.write(f"現在の構成: 数値列 {len(numeric_cols)}個, カテゴリ列 {len(categorical_cols)}個, テキスト列 {len(text_cols)}個, 日時列 {len(datetime_cols)}個")
    
    # Feature engineering options
    feature_options = st.multiselect(
        '作成する特徴量を選択:',
        [
            'OneHot Encoding (カテゴリ→数値)',
            'テキスト特徴量 (長さ、単語数など)',
            '日時特徴量 (年、月、曜日など)', 
            '数値交互作用 (掛け算、割り算など)',
            '数値ビニング (数値→カテゴリ)',
            '統計的特徴量 (標準化、ランク付けなど)'
        ]
    )
    
    new_features_df = sampled_pd.copy()
    feature_info = {}
    
    if 'OneHot Encoding (カテゴリ→数値)' in feature_options and categorical_cols:
        st.write('### OneHot Encoding')
        selected_cat_cols = st.multiselect('OneHot化するカテゴリ列:', categorical_cols, default=categorical_cols)
        max_categories = st.slider('カテゴリあたりの最大ダミー変数数:', min_value=3, max_value=20, value=10)
        
        if selected_cat_cols:
            encoded_df, encoding_info = create_onehot_features(sampled_pd, selected_cat_cols, max_categories)
            
            # Add new columns to the main dataframe
            new_cols = []
            for col_info in encoding_info.values():
                new_cols.extend(col_info['encoded_cols'])
            
            new_features_df = pd.concat([new_features_df, encoded_df[new_cols]], axis=1)
            feature_info['onehot'] = encoding_info
            
            st.write(f'追加された列数: {len(new_cols)}')
            st.write('エンコーディング情報:', encoding_info)
    
    if 'テキスト特徴量 (長さ、単語数など)' in feature_options and text_cols:
        st.write('### テキスト特徴量')
        selected_text_cols = st.multiselect('分析するテキスト列:', text_cols, default=text_cols)
        
        if selected_text_cols:
            text_features = create_numeric_from_text(sampled_pd, selected_text_cols)
            new_features_df = pd.concat([new_features_df, text_features], axis=1)
            feature_info['text'] = list(text_features.columns)
            st.write(f'追加されたテキスト特徴量: {len(text_features.columns)}個')
    
    if '日時特徴量 (年、月、曜日など)' in feature_options and datetime_cols:
        st.write('### 日時特徴量')
        selected_dt_cols = st.multiselect('分析する日時列:', datetime_cols, default=datetime_cols)
        
        if selected_dt_cols:
            dt_features = create_datetime_features(sampled_pd, selected_dt_cols)
            new_features_df = pd.concat([new_features_df, dt_features], axis=1)
            feature_info['datetime'] = list(dt_features.columns)
            st.write(f'追加された日時特徴量: {len(dt_features.columns)}個')
    
    if '数値交互作用 (掛け算、割り算など)' in feature_options and len(numeric_cols) >= 2:
        st.write('### 数値交互作用特徴量')
        max_interactions = st.slider('交互作用を作る最大列数:', min_value=2, max_value=min(10, len(numeric_cols)), value=min(5, len(numeric_cols)))
        
        interaction_features = create_interaction_features(sampled_pd, numeric_cols, max_interactions)
        new_features_df = pd.concat([new_features_df, interaction_features], axis=1)
        feature_info['interactions'] = list(interaction_features.columns)
        st.write(f'追加された交互作用特徴量: {len(interaction_features.columns)}個')
    
    if '数値ビニング (数値→カテゴリ)' in feature_options and numeric_cols:
        st.write('### 数値ビニング')
        selected_num_cols = st.multiselect('ビニングする数値列:', numeric_cols)
        n_bins = st.slider('ビン数:', min_value=3, max_value=10, value=5)
        
        if selected_num_cols:
            binned_features = create_binned_features(sampled_pd, selected_num_cols, n_bins)
            new_features_df = pd.concat([new_features_df, binned_features], axis=1)
            feature_info['binned'] = list(binned_features.columns)
            st.write(f'追加されたビニング特徴量: {len(binned_features.columns)}個')
    
    if '統計的特徴量 (標準化、ランク付けなど)' in feature_options and numeric_cols:
        st.write('### 統計的特徴量')
        selected_stat_cols = st.multiselect('統計処理する数値列:', numeric_cols)
        
        if selected_stat_cols:
            stat_features = {}
            for col in selected_stat_cols:
                # Standardization
                stat_features[f'{col}_std'] = (sampled_pd[col] - sampled_pd[col].mean()) / sampled_pd[col].std()
                # Rank
                stat_features[f'{col}_rank'] = sampled_pd[col].rank()
                # Log transform (with protection)
                stat_features[f'{col}_log'] = np.log1p(np.abs(sampled_pd[col]))
                # Square root
                stat_features[f'{col}_sqrt'] = np.sqrt(np.abs(sampled_pd[col]))
            
            stat_df = pd.DataFrame(stat_features)
            new_features_df = pd.concat([new_features_df, stat_df], axis=1)
            feature_info['statistical'] = list(stat_df.columns)
            st.write(f'追加された統計的特徴量: {len(stat_df.columns)}個')
    
    # Show results
    if len(new_features_df.columns) > len(sampled_pd.columns):
        st.success(f'特徴量エンジニアリング完了！ {len(sampled_pd.columns)} → {len(new_features_df.columns)} 列')
        
        # Update the dataframe in session state for other tabs to use
        st.session_state['engineered_df'] = new_features_df
        st.session_state['feature_info'] = feature_info
        
        # Show new feature statistics
        new_cols = [col for col in new_features_df.columns if col not in sampled_pd.columns]
        if st.checkbox('新しい特徴量を表示', value=False):
            st.dataframe(new_features_df[new_cols].head(100))
        
        # Download engineered features
        csv_engineered = new_features_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            'エンジニアリング済みデータをダウンロード',
            data=csv_engineered,
            file_name=f'engineered_{selected_file}',
            mime='text/csv'
        )
        
        # Update types for new features
        if st.button('新しい特徴量で型推定を更新'):
            # Re-infer types for the new dataframe
            new_types = infer_column_types(new_features_df)
            st.session_state['engineered_types'] = new_types
            st.success('型推定を更新しました')
    else:
        st.info('特徴量を選択して作成してください')

# -------- Stat Tests --------
with tabs[3]:
    st.subheader('統計的検定 & 推奨検定')
    
    # Use engineered dataframe if available
    analysis_df = st.session_state.get('engineered_df', sampled_pd)
    analysis_types = st.session_state.get('engineered_types', types)
    
    # Get column types from the analysis dataframe
    numeric_cols = [c for c,t in analysis_types.items() if t=='numeric']
    categorical_cols = [c for c,t in analysis_types.items() if t=='categorical']
    
    st.write(f'解析対象: {analysis_df.shape[1]}列 (数値: {len(numeric_cols)}, カテゴリ: {len(categorical_cols)})')
    
    # Re-compute basic stats for analysis dataframe
    analysis_stats = compute_basic_stats(analysis_df, analysis_types)
    
    suggestions = suggest_tests_for_columns(analysis_types, analysis_stats)
    st.write('おすすめの検定（自動提案）')
    for s in suggestions[:10]:
        st.write(s)
    st.write('---')
    
    # Get available columns for each type
    if not numeric_cols:
        st.warning('数値列が見つかりません。特徴量エンジニアリングタブでOneHotエンコーディングやテキスト特徴量を作成してください。')
    if not categorical_cols:
        st.warning('カテゴリ列が見つかりません。特徴量エンジニアリングタブで数値ビニングを試してください。')
    
    # manual test execution
    test_type = st.selectbox('検定タイプ', ['t-test (indep)', 'paired t-test', 'ANOVA', 'Mann-Whitney U', 'Chi-square'])
    
    if test_type in ['t-test (indep)','paired t-test','ANOVA','Mann-Whitney U']:
        # Only show options if numeric and categorical columns exist
        if not numeric_cols:
            st.info('💡 ヒント: OneHotエンコーディングやテキスト特徴量で数値列を作成できます')
        elif not categorical_cols:
            st.info('💡 ヒント: 数値ビニングでカテゴリ列を作成できます')
        else:
            num = st.selectbox('数値列 (従属)', options=numeric_cols, key='numeric_col_selection_eng')
            grp = st.selectbox('グループ列 (独立変数)', options=categorical_cols, key='group_col_selection_eng')
            
            if st.button('検定を実行') and num is not None and grp is not None:
                res = None
                if test_type=='t-test (indep)':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num]
                        b = grp_df[grp_df[grp]==groups[1]][num]
                        r = stats.ttest_ind(a,b, equal_var=False, nan_policy='omit')
                        res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='paired t-test':
                    st.info('対応t検定はデータを整列させたペアが必要です。サンプルはペアがない場合は最初のNを使います')
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num].values
                        b = grp_df[grp_df[grp]==groups[1]][num].values
                        m = min(len(a), len(b))
                        if m == 0:
                            st.error('データが不十分です')
                        else:
                            r = stats.ttest_rel(a[:m], b[:m], nan_policy='omit')
                            res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='ANOVA':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    arrays = [grp_df[grp_df[grp]==g][num].values for g in groups]
                    r = stats.f_oneway(*arrays)
                    res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='Mann-Whitney U':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num]
                        b = grp_df[grp_df[grp]==groups[1]][num]
                        r = stats.mannwhitneyu(a,b, alternative='two-sided')
                        res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                if res:
                    st.json(res)
                    st.write('p < 0.05 -> 帰無仮説を棄却' if res.get('pvalue',1) < 0.05 else '帰無仮説を採択')
    else:
        # Chi-square test
        non_numeric_cols = [c for c,t in analysis_types.items() if t!='numeric']
        if len(non_numeric_cols) < 2:
            st.warning('カイ二乗検定には2つ以上のカテゴリ列が必要です。')
            st.info('💡 ヒント: 数値ビニングでカテゴリ列を作成できます')
        else:
            c1 = st.selectbox('カテゴリ列1', options=non_numeric_cols, key='cat1_selection_eng')
            c2 = st.selectbox('カテゴリ列2', options=[c for c in non_numeric_cols if c!=c1], key='cat2_selection_eng')
            if st.button('カイ二乗検定実行') and c1 is not None and c2 is not None:
                try:
                    ct = pd.crosstab(analysis_df[c1], analysis_df[c2])
                    chi2, p, dof, ex = stats.chi2_contingency(ct)
                    st.json({'chi2':float(chi2), 'pvalue':float(p), 'dof':int(dof)})
                    st.write('p < 0.05 -> 帰無仮説を棄却 (2つの変数に関連あり)' if p < 0.05 else '帰無仮説を採択 (2つの変数に関連なし)')
                except Exception as e:
                    st.error(f'カイ二乗検定でエラーが発生しました: {e}')

# -------- PCA & Clustering --------
with tabs[4]:
    st.subheader('PCA & Clustering')
    
    # Use engineered dataframe if available
    analysis_df = st.session_state.get('engineered_df', sampled_pd)
    analysis_types = st.session_state.get('engineered_types', types)
    numeric_cols = [c for c,t in analysis_types.items() if t=='numeric']
    
    if len(numeric_cols) < 2:
        st.info('2つ以上の数値列が必要です')
        st.info('💡 ヒント: 特徴量エンジニアリングタブでOneHotエンコーディングやテキスト特徴量を作成してください')
    else:
        st.write(f'使用可能な数値列: {len(numeric_cols)}個')
        
        # Column selection for PCA
        selected_pca_cols = st.multiselect(
            'PCAに使用する列を選択:', 
            numeric_cols, 
            default=numeric_cols[:min(10, len(numeric_cols))]
        )
        
        if selected_pca_cols and len(selected_pca_cols) >= 2:
            n_comp = st.slider('主成分数', min_value=2, max_value=min(6, len(selected_pca_cols)), value=min(3, len(selected_pca_cols)))
            pc_res = run_pca(analysis_df, selected_pca_cols, n_comp)
            if pc_res is not None:
                comp_df, var = pc_res
                comp_df_display = comp_df.copy()
                comp_df_display['index'] = np.arange(len(comp_df_display))
                
                # PCA scatter plot with better visualization
                fig = px.scatter(
                    comp_df_display, 
                    x='PC1', y='PC2', 
                    title=f'PCA: PC1 ({var[0]:.1%}) vs PC2 ({var[1]:.1%})',
                    labels={'PC1': f'PC1 ({var[0]:.1%})', 'PC2': f'PC2 ({var[1]:.1%})'}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write('各主成分の寄与率:', [f'PC{i+1}: {float(v):.1%}' for i, v in enumerate(var)])
                st.write('累積寄与率:', f'{sum(var):.1%}')
                
            # Clustering
            st.write('### クラスタリング')
            k = st.slider('クラスタ数 (KMeans)', min_value=2, max_value=10, value=3)
            cluster_cols = st.multiselect(
                'クラスタリングに使用する列:', 
                numeric_cols, 
                default=selected_pca_cols[:min(5, len(selected_pca_cols))]
            )
            
            if cluster_cols:
                labels = run_kmeans(analysis_df, cluster_cols, k)
                if labels is not None:
                    # Create visualization dataframe
                    X = analysis_df[cluster_cols].dropna().copy()
                    X['cluster'] = labels
                    
                    # Scatter matrix for clusters
                    if len(cluster_cols) >= 2:
                        figc = px.scatter_matrix(
                            X.sample(n=min(2000, X.shape[0])), 
                            dimensions=cluster_cols[:4], 
                            color='cluster',
                            title=f'クラスタ分析 (k={k})'
                        )
                        st.plotly_chart(figc, use_container_width=True)
                        
                        # Cluster statistics
                        cluster_stats = X.groupby('cluster')[cluster_cols].mean()
                        st.write('### クラスタ別統計')
                        st.dataframe(cluster_stats)
                        
                        # Add cluster labels to session state
                        analysis_df_with_clusters = analysis_df.copy()
                        analysis_df_with_clusters['cluster'] = np.nan
                        analysis_df_with_clusters.loc[X.index, 'cluster'] = labels
                        st.session_state['clustered_df'] = analysis_df_with_clusters

# ======== AutoML Tab ========
with tabs[5]:
    st.subheader('🤖 AutoML - 自動機械学習')
    
    # Use engineered dataframe if available
    ml_df = st.session_state.get('engineered_df', sampled_pd)
    
    st.write(f'データセット: {ml_df.shape[0]}行, {ml_df.shape[1]}列')
    
    # Target column selection
    st.write('### 1. ターゲット列の選択')
    target_col = st.selectbox(
        'ターゲット列（予測したい変数）を選択:',
        options=['選択してください'] + list(ml_df.columns),
        key='target_selection'
    )
    
    if target_col == '選択してください':
        st.info('ターゲット列を選択してAutoMLを開始してください')
    else:
        # Display target information
        st.write('ターゲット列の統計情報:')
        target_series = ml_df[target_col]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('データ数', target_series.notna().sum())
        with col2:
            st.metric('欠損値', target_series.isna().sum())
        with col3:
            st.metric('ユニーク値数', target_series.nunique())
        with col4:
            unique_ratio = target_series.nunique() / len(target_series)
            st.metric('ユニーク率', f'{unique_ratio:.3f}')
        
        # Auto-detect task type
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_ratio < 0.05 or target_series.nunique() <= 20:
                detected_task = 'classification'
            else:
                detected_task = 'regression'
        else:
            detected_task = 'classification'
        
        task_type = st.selectbox(
            'タスクタイプ:',
            options=['auto', 'classification', 'regression'],
            help=f'自動検出: {detected_task}'
        )
        
        if task_type == 'auto':
            task_type = detected_task
        
        st.info(f'選択されたタスク: **{task_type}**')
        
        # Model selection
        st.write('### 2. モデル選択')
        available_models = ['LightGBM', 'XGBoost', 'RandomForest']
        selected_models = st.multiselect(
            '使用するモデル:',
            options=available_models,
            default=available_models
        )
        
        if not selected_models:
            st.warning('少なくとも1つのモデルを選択してください')
        else:
            # Cross-validation settings
            st.write('### 3. クロスバリデーション設定')
            col1, col2 = st.columns(2)
            
            with col1:
                cv_type = st.selectbox(
                    'CV手法:',
                    options=['StratifiedKFold', 'KFold', 'TimeSeriesSplit'],
                    help='StratifiedKFold: 分類用, KFold: 回帰用, TimeSeriesSplit: 時系列用'
                )
            
            with col2:
                n_splits = st.slider('分割数:', min_value=3, max_value=10, value=5)
            
            # Optuna settings
            st.write('### 4. ハイパーパラメータ最適化設定')
            col1, col2 = st.columns(2)
            
            with col1:
                n_trials = st.slider('試行回数:', min_value=10, max_value=500, value=100)
            
            with col2:
                timeout_minutes = st.slider('制限時間 (分):', min_value=1, max_value=60, value=10)
            
            # Feature selection
            st.write('### 5. 特徴量選択')
            feature_cols = [col for col in ml_df.columns if col != target_col]
            
            if st.checkbox('すべての特徴量を使用', value=True):
                selected_features = feature_cols
            else:
                selected_features = st.multiselect(
                    '使用する特徴量:',
                    options=feature_cols,
                    default=feature_cols[:20]  # Limit default selection
                )
            
            st.write(f'選択された特徴量数: {len(selected_features)}')
            
            # Start AutoML
            if st.button('🚀 AutoML実行', type='primary'):
                if not selected_features:
                    st.error('特徴量を選択してください')
                else:
                    with st.spinner('AutoMLを実行中...'):
                        try:
                            # Preprocessing
                            st.info('前処理中...')
                            ml_subset = ml_df[selected_features + [target_col]].copy()
                            X, y, preprocessing_info = preprocess_for_automl(ml_subset, target_col, task_type)
                            
                            st.success(f'前処理完了: {X.shape[1]}特徴量, {len(y)}サンプル')
                            
                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42,
                                stratify=y if task_type == 'classification' else None
                            )
                            
                            # Store results
                            automl_results = {}
                            
                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Train each model
                            for i, model_name in enumerate(selected_models):
                                status_text.text(f'最適化中: {model_name}...')
                                progress_bar.progress((i) / len(selected_models))
                                
                                start_time = time.time()
                                
                                # Run optimization
                                result = run_automl_optimization(
                                    X_train, y_train, model_name, task_type,
                                    cv_type, n_splits, n_trials, timeout_minutes
                                )
                                
                                training_time = time.time() - start_time
                                
                                # Evaluate on test set
                                test_results = evaluate_model(result['model'], X_test, y_test, task_type)
                                
                                # Compute feature importance
                                feature_importance = compute_feature_importance_automl(
                                    result['model'], X.columns
                                )
                                
                                # Store results
                                automl_results[model_name] = {
                                    'model': result['model'],
                                    'best_params': result['best_params'],
                                    'cv_score': result['best_score'],
                                    'test_results': test_results,
                                    'feature_importance': feature_importance,
                                    'training_time': training_time,
                                    'study': result['study']
                                }
                            
                            progress_bar.progress(1.0)
                            status_text.text('完了!')
                            
                            # Store in session state
                            st.session_state['automl_results'] = automl_results
                            st.session_state['automl_task_type'] = task_type
                            st.session_state['automl_X_test'] = X_test
                            st.session_state['automl_y_test'] = y_test
                            st.session_state['automl_preprocessing'] = preprocessing_info
                            
                            st.success('AutoML完了!')
                            
                        except Exception as e:
                            st.error(f'AutoMLでエラーが発生: {str(e)}')
                            import traceback
                            st.code(traceback.format_exc())

# Display AutoML results if available
if 'automl_results' in st.session_state:
    st.write('---')
    st.write('### 🏆 モデル比較結果')
    
    results = st.session_state['automl_results']
    task_type = st.session_state['automl_task_type']
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'CV Score': f"{result['cv_score']:.4f}",
            'Training Time (s)': f"{result['training_time']:.1f}"
        }
        
        if task_type == 'classification':
            row['Test Accuracy'] = f"{result['test_results']['accuracy']:.4f}"
            row['Test F1'] = f"{result['test_results']['f1']:.4f}"
            if 'roc_auc' in result['test_results']:
                row['Test ROC-AUC'] = f"{result['test_results']['roc_auc']:.4f}"
        else:
            row['Test RMSE'] = f"{result['test_results']['rmse']:.4f}"
            row['Test MAE'] = f"{result['test_results']['mae']:.4f}"
            row['Test R²'] = f"{result['test_results']['r2']:.4f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model selection for detailed analysis
    st.write('### 📊 詳細分析')
    selected_model = st.selectbox(
        '詳細分析するモデルを選択:',
        options=list(results.keys())
    )
    
    if selected_model:
        result = results[selected_model]
        
        # Performance metrics
        st.write(f'#### {selected_model} - 性能指標')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('**クロスバリデーション結果**')
            st.metric('CV Score', f"{result['cv_score']:.4f}")
            
            st.write('**最適化されたパラメータ**')
            st.json(result['best_params'])
        
        with col2:
            st.write('**テストセット結果**')
            test_results = result['test_results']
            
            if task_type == 'classification':
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric('Accuracy', f"{test_results['accuracy']:.4f}")
                    st.metric('F1 Score', f"{test_results['f1']:.4f}")
                with col2_2:
                    if 'roc_auc' in test_results:
                        st.metric('ROC-AUC', f"{test_results['roc_auc']:.4f}")
                    if 'pr_auc' in test_results:
                        st.metric('PR-AUC', f"{test_results['pr_auc']:.4f}")
                
                # Confusion Matrix
                if 'confusion_matrix' in test_results:
                    st.write('**混同行列**')
                    cm = test_results['confusion_matrix']
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        title='Confusion Matrix',
                        labels=dict(x="Predicted", y="Actual")
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            else:  # regression
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric('RMSE', f"{test_results['rmse']:.4f}")
                    st.metric('MAE', f"{test_results['mae']:.4f}")
                with col2_2:
                    st.metric('R² Score', f"{test_results['r2']:.4f}")
                
                # Actual vs Predicted plot
                y_test = st.session_state['automl_y_test']
                y_pred = test_results['predictions']
                
                pred_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                
                fig_pred = px.scatter(
                    pred_df, 
                    x='Actual', 
                    y='Predicted',
                    title='Actual vs Predicted',
                    trendline='ols'
                )
                # Add perfect prediction line
                min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                fig_pred.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=min_val, y0=min_val, x1=max_val, y1=max_val
                )
                st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        st.write('#### 📈 特徴量重要度')
        if result['feature_importance'] is not None:
            importance_df = result['feature_importance'].head(20)
            
            fig_importance = px.bar(
                importance_df,
                y='feature',
                x='importance',
                orientation='h',
                title=f'{selected_model} - Top 20 Feature Importance'
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Download feature importance
            csv_importance = importance_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '特徴量重要度をダウンロード (CSV)',
                data=csv_importance,
                file_name=f'{selected_model}_feature_importance.csv',
                mime='text/csv'
            )
        else:
            st.info('特徴量重要度は利用できません')
        
        # SHAP Analysis
        st.write('#### 🔍 SHAP解釈可能性分析')
        
        if st.button('SHAP分析を実行'):
            with st.spinner('SHAP値を計算中...'):
                X_test = st.session_state['automl_X_test']
                shap_values, explainer, X_sample = compute_shap_values(
                    result['model'], X_test, max_samples=100
                )
                
                if shap_values is not None:
                    st.success('SHAP分析完了!')
                    
                    # Handle different SHAP value shapes
                    # For multi-class classification, shap_values might be 3D
                    if len(shap_values.shape) == 3:
                        # Multi-class case: take the values for the first class or average
                        st.info(f'多クラス分類が検出されました。形状: {shap_values.shape}')
                        if task_type == 'classification':
                            # Use values for class 1 (positive class) or average across classes
                            if shap_values.shape[0] >= 2:
                                shap_values_2d = shap_values[1]  # Positive class
                                st.info('陽性クラスのSHAP値を使用します')
                            else:
                                shap_values_2d = shap_values[0]  # First class
                                st.info('第1クラスのSHAP値を使用します')
                        else:
                            # For regression, average across the last dimension if needed
                            shap_values_2d = np.mean(shap_values, axis=0) if shap_values.shape[0] > 1 else shap_values[0]
                    elif len(shap_values.shape) == 2:
                        # Standard case: 2D array
                        shap_values_2d = shap_values
                    else:
                        st.error(f'予期しないSHAP値の形状: {shap_values.shape}')
                        st.stop()
                    
                    # Ensure we have the right shape
                    if shap_values_2d.shape[0] != len(X_sample) or shap_values_2d.shape[1] != len(X_sample.columns):
                        st.error(f'SHAP値の形状が一致しません。期待: {(len(X_sample), len(X_sample.columns))}, 実際: {shap_values_2d.shape}')
                        st.stop()
                    
                    # SHAP Summary Plot (using plotly)
                    shap_df = pd.DataFrame(shap_values_2d, columns=X_sample.columns)
                    
                    # Feature importance based on mean absolute SHAP values
                    shap_importance = pd.DataFrame({
                        'feature': shap_df.columns,
                        'mean_abs_shap': np.abs(shap_df).mean().values
                    }).sort_values('mean_abs_shap', ascending=False).head(15)
                    
                    fig_shap = px.bar(
                        shap_importance,
                        y='feature',
                        x='mean_abs_shap',
                        orientation='h',
                        title='SHAP Feature Importance (Mean |SHAP value|)'
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # SHAP values for individual prediction
                    st.write('**個別予測のSHAP値**')
                    sample_idx = st.slider(
                        'サンプル選択:', 
                        min_value=0, 
                        max_value=len(X_sample)-1, 
                        value=0
                    )
                    
                    sample_shap = shap_values_2d[sample_idx]
                    sample_features = X_sample.iloc[sample_idx]
                    
                    sample_shap_df = pd.DataFrame({
                        'feature': X_sample.columns,
                        'feature_value': sample_features.values,
                        'shap_value': sample_shap
                    }).sort_values('shap_value', key=abs, ascending=False).head(15)
                    
                    fig_sample = px.bar(
                        sample_shap_df,
                        y='feature',
                        x='shap_value',
                        orientation='h',
                        title=f'SHAP Values for Sample {sample_idx}',
                        color='shap_value',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig_sample, use_container_width=True)
                    
                    # Store SHAP results
                    st.session_state[f'shap_values_{selected_model}'] = shap_values_2d
                    st.session_state[f'shap_sample_{selected_model}'] = X_sample
        
        # Model download
        st.write('#### 💾 モデル保存')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Save model
            model_bytes = BytesIO()
            joblib.dump(result['model'], model_bytes)
            model_bytes.seek(0)
            
            st.download_button(
                'モデルをダウンロード (.pkl)',
                data=model_bytes.getvalue(),
                file_name=f'{selected_model}_model.pkl',
                mime='application/octet-stream'
            )
        
        with col2:
            # Save preprocessing pipeline
            preprocessing_bytes = BytesIO()
            joblib.dump(st.session_state['automl_preprocessing'], preprocessing_bytes)
            preprocessing_bytes.seek(0)
            
            st.download_button(
                '前処理パイプラインをダウンロード (.pkl)',
                data=preprocessing_bytes.getvalue(),
                file_name='preprocessing_pipeline.pkl',
                mime='application/octet-stream'
            )

# -------- Merge & Export --------
with tabs[6]:
    st.subheader('ファイルマージ & エクスポート')
    all_files = list(file_store.keys())
    
    # Export engineered features
    if 'engineered_df' in st.session_state:
        st.write('### エンジニアリング済みデータのエクスポート')
        engineered_df = st.session_state['engineered_df']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('元の列数', len(sampled_pd.columns))
        with col2:
            st.metric('新しい列数', len(engineered_df.columns))
            
        # Show feature creation summary
        if 'feature_info' in st.session_state:
            feature_info = st.session_state['feature_info']
            st.write('### 作成された特徴量の詳細')
            for feature_type, info in feature_info.items():
                if feature_type == 'onehot':
                    for col, details in info.items():
                        st.write(f"**{col}** → {len(details['encoded_cols'])}個のOneHot特徴量")
                else:
                    st.write(f"**{feature_type}**: {len(info)}個の特徴量")
        
        # Download options
        csv_engineered = engineered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '🔽 エンジニアリング済みデータをダウンロード (CSV)',
            data=csv_engineered,
            file_name=f'engineered_{selected_file}',
            mime='text/csv'
        )
        
        # Export with clusters if available
        if 'clustered_df' in st.session_state:
            clustered_df = st.session_state['clustered_df']
            csv_clustered = clustered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '🔽 クラスタ情報付きデータをダウンロード (CSV)',
                data=csv_clustered,
                file_name=f'clustered_{selected_file}',
                mime='text/csv'
            )
    
    st.write('---')
    
    # File merging
    if len(all_files) < 2:
        st.info('2つ以上のファイルをアップロードするとマージできます')
    else:
        st.write('### ファイルマージ')
        left = st.selectbox('Left file', options=all_files)
        right = st.selectbox('Right file', options=[f for f in all_files if f!=left])
        l_pl, l_pd = file_store[left]
        r_pl, r_pd = file_store[right]
        common = list(set(l_pl.columns) & set(r_pl.columns))
        st.write('共通列候補:', common)
        left_keys = st.multiselect('Left key(s)', options=list(l_pl.columns), default=common[:1] if common else [])
        right_keys = st.multiselect('Right key(s)', options=list(r_pl.columns), default=common[:1] if common else [])
        how = st.selectbox('Join type', ['inner','left','right','outer'])
        if st.button('マージ実行'):
            if not left_keys or not right_keys or len(left_keys)!=len(right_keys):
                st.error('左と右で対応するキーを同数選択してください')
            else:
                with st.spinner('マージ中...'):
                    try:
                        merged = l_pd.merge(r_pd, left_on=left_keys, right_on=right_keys, how=how, suffixes=('_l','_r'))
                        st.success('マージ完了')
                        st.dataframe(merged.head(500))
                        csv = merged.to_csv(index=False).encode('utf-8')
                        st.download_button('ダウンロード (merged.csv)', data=csv, file_name='merged.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f'マージでエラー: {e}')

# -------- Report --------
with tabs[7]:
    st.subheader('レポート生成')
    
    # Use engineered dataframe if available for report
    report_df = st.session_state.get('engineered_df', sampled_pd)
    report_types = st.session_state.get('engineered_types', types)
    
    title = st.text_input('レポートタイトル', value=f'EDA + AutoML Report - {selected_file}')
    
    # Enhanced description with feature engineering and AutoML info
    default_desc = 'このレポートは EDA Studio Pro により自動生成されました。'
    
    if 'feature_info' in st.session_state:
        feature_info = st.session_state['feature_info']
        feature_count = sum(len(info) if isinstance(info, list) else len(info) for info in feature_info.values())
        default_desc += f' {feature_count}個の新しい特徴量が作成されました。'
    
    if 'automl_results' in st.session_state:
        automl_results = st.session_state['automl_results']
        best_model = max(automl_results.items(), key=lambda x: x[1]['cv_score'])
        default_desc += f' AutoMLにより{len(automl_results)}個のモデルが比較され、最良モデルは{best_model[0]}でした（CV Score: {best_model[1]["cv_score"]:.4f}）。'
    
    desc = st.text_area('説明 (レポート本文に入る要約)', value=default_desc)
    
    # Plot selection for report
    plot_options = st.multiselect(
        'レポートに含める図表:',
        [
            '数値列の分布',
            '相関行列',
            'PCA結果',
            'クラスタリング結果',
            'AutoML性能比較',
            '特徴量重要度',
            'SHAP分析結果'
        ],
        default=['数値列の分布', '相関行列', 'AutoML性能比較']
    )
    
    imgs = []
    
    if st.button('レポート用プロットを生成'): 
        with st.spinner('プロット生成中...'):
            numeric_cols = [c for c,t in report_types.items() if t=='numeric']
            
            if '数値列の分布' in plot_options and numeric_cols:
                top_num = numeric_cols[0]
                fig1 = px.histogram(report_df, x=top_num, nbins=60, title=f'Distribution: {top_num}')
                img1 = plotly_to_png(fig1)
                imgs.append((img1, f'Distribution of {top_num}'))
            
            if '相関行列' in plot_options and len(numeric_cols) >= 2:
                corr = report_df[numeric_cols[:10]].corr()  # Limit to top 10 for readability
                fig2 = px.imshow(corr, text_auto='.2f', title='Correlation matrix')
                img2 = plotly_to_png(fig2)
                imgs.append((img2, 'Correlation matrix (top 10 numeric features)'))
            
            if 'PCA結果' in plot_options and len(numeric_cols) >= 2:
                pc_res = run_pca(report_df, numeric_cols[:10], 3)
                if pc_res is not None:
                    comp_df, var = pc_res
                    comp_df_display = comp_df.copy()
                    comp_df_display['index'] = np.arange(len(comp_df_display))
                    fig3 = px.scatter(
                        comp_df_display, 
                        x='PC1', y='PC2', 
                        title=f'PCA: PC1 ({var[0]:.1%}) vs PC2 ({var[1]:.1%})'
                    )
                    img3 = plotly_to_png(fig3)
                    imgs.append((img3, 'Principal Component Analysis'))
            
            if 'クラスタリング結果' in plot_options and 'clustered_df' in st.session_state:
                clustered_df = st.session_state['clustered_df']
                cluster_col = 'cluster'
                if cluster_col in clustered_df.columns:
                    cluster_summary = clustered_df.groupby(cluster_col).size().reset_index(name='count')
                    fig4 = px.bar(cluster_summary, x=cluster_col, y='count', title='Cluster Distribution')
                    img4 = plotly_to_png(fig4)
                    imgs.append((img4, 'Cluster analysis results'))
            
            if 'AutoML性能比較' in plot_options and 'automl_results' in st.session_state:
                automl_results = st.session_state['automl_results']
                task_type = st.session_state['automl_task_type']
                
                # Create performance comparison chart
                model_names = list(automl_results.keys())
                cv_scores = [result['cv_score'] for result in automl_results.values()]
                
                fig5 = px.bar(
                    x=model_names,
                    y=cv_scores,
                    title='AutoML Model Performance Comparison (CV Score)'
                )
                img5 = plotly_to_png(fig5)
                imgs.append((img5, 'AutoML model performance comparison'))
            
            if '特徴量重要度' in plot_options and 'automl_results' in st.session_state:
                # Use best model's feature importance
                automl_results = st.session_state['automl_results']
                best_model = max(automl_results.items(), key=lambda x: x[1]['cv_score'])
                
                if best_model[1]['feature_importance'] is not None:
                    importance_df = best_model[1]['feature_importance'].head(15)
                    fig6 = px.bar(
                        importance_df,
                        y='feature',
                        x='importance',
                        orientation='h',
                        title=f'{best_model[0]} - Top 15 Feature Importance'
                    )
                    img6 = plotly_to_png(fig6)
                    imgs.append((img6, f'Feature importance from best model: {best_model[0]}'))
            
            st.success('プロット生成完了')
            st.session_state['report_imgs'] = imgs
    
    # Store images in session state to persist them
    if 'report_imgs' in st.session_state:
        imgs = st.session_state['report_imgs']
        
    if imgs:
        st.write(f'生成されたプロット数: {len(imgs)}')
        
        if st.button('PDFレポートを作成'):
            with st.spinner('PDF作成中...'):
                pdf_bytes = create_pdf_report(title, desc, imgs)
                st.download_button('PDF をダウンロード', data=pdf_bytes, file_name='eda_automl_report.pdf', mime='application/pdf')
    
    # Feature engineering + AutoML summary
    if 'feature_info' in st.session_state or 'automl_results' in st.session_state:
        st.write('---')
        st.write('### 解析サマリ')
        
        summary_text = "# EDA + AutoML Analysis Summary\n\n"
        
        # Feature engineering summary
        if 'feature_info' in st.session_state:
            feature_info = st.session_state['feature_info']
            
            summary_text += "## Feature Engineering Summary\n\n"
            for feature_type, info in feature_info.items():
                if feature_type == 'onehot':
                    summary_text += f"### OneHot Encoding\n"
                    for col, details in info.items():
                        summary_text += f"- {col}: {details['original_unique']} categories → {len(details['encoded_cols'])} binary features\n"
                else:
                    feature_count = len(info) if isinstance(info, list) else len(info)
                    summary_text += f"### {feature_type.title()}: {feature_count} features created\n"
        
        # AutoML summary
        if 'automl_results' in st.session_state:
            automl_results = st.session_state['automl_results']
            task_type = st.session_state['automl_task_type']
            
            summary_text += f"\n## AutoML Results Summary\n\n"
            summary_text += f"**Task Type**: {task_type}\n"
            summary_text += f"**Models Compared**: {len(automl_results)}\n\n"
            
            # Best model
            best_model = max(automl_results.items(), key=lambda x: x[1]['cv_score'])
            summary_text += f"**Best Model**: {best_model[0]}\n"
            summary_text += f"**Best CV Score**: {best_model[1]['cv_score']:.4f}\n\n"
            
            # All model results
            summary_text += "### Model Comparison\n\n"
            for model_name, result in automl_results.items():
                summary_text += f"**{model_name}**\n"
                summary_text += f"- CV Score: {result['cv_score']:.4f}\n"
                summary_text += f"- Training Time: {result['training_time']:.1f}s\n"
                
                if task_type == 'classification':
                    summary_text += f"- Test Accuracy: {result['test_results']['accuracy']:.4f}\n"
                    summary_text += f"- Test F1: {result['test_results']['f1']:.4f}\n"
                else:
                    summary_text += f"- Test RMSE: {result['test_results']['rmse']:.4f}\n"
                    summary_text += f"- Test R²: {result['test_results']['r2']:.4f}\n"
                
                summary_text += "\n"
        
        st.markdown(summary_text)
        
        # Download comprehensive report
        summary_bytes = summary_text.encode('utf-8')
        st.download_button(
            '完全解析レポートをダウンロード (Markdown)',
            data=summary_bytes,
            file_name='complete_analysis_report.md',
            mime='text/markdown'
        )

st.success('EDA Studio Pro + AutoML を起動しました')