import re
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from xgboost import XGBClassifier


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake case.

    Parameters
    ----------
    name : str
        The string to convert.

    Returns
    -------
    str
        The converted string.
    """
    name = name.replace(' ', '_')
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower().strip()

def build_column_transformer_for_df(train_x: pd.DataFrame) -> ColumnTransformer:
    """Builds a column transformer for a pandas dataframe."""
    # Get the categorical and numerical columns
    categorical_columns = train_x.select_dtypes(
        include='object').columns.to_list()
    numerical_columns = train_x.select_dtypes(
        include='number').columns.to_list()

    num_prep = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_prep = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ('num', num_prep, numerical_columns),
        ('cat', cat_prep, categorical_columns)
    ])

    return transformer


def build_sklearn_pipeline(df: pd.DataFrame, y_col_name: str, model_name: str, model: object) -> Pipeline:
    """Builds a sklearn pipeline for churn prediction."""
    # Define the steps

    transformer = build_column_transformer_for_df(df.drop(y_col_name, axis=1))

    steps = [
        ('preprocessor', transformer),
        ('pca', PCA()),
        (model_name, model)
    ]
    # Create the pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


def sklearn_gridsearch_using_pipeline(train: pd.DataFrame, y_col_name: str, model_name: str, model: object, fit_le: LabelEncoder, param_grid_model: dict, n_folds: int = 5,) -> GridSearchCV:
    """Performs a grid search using a sklearn pipeline."""
    # Get the pipeline
    pipeline = build_sklearn_pipeline(
        train, y_col_name=y_col_name, model=model, model_name=model_name)

    # define stratiefied shuffle split:
    sss = StratifiedShuffleSplit(
        n_splits=n_folds, test_size=0.2, random_state=0)

    # Define the hyperparameter grid
    param_grid = param_grid_model
    param_grid["pca__n_components"] = [15, 20, 25, 30, 35, 50, 65]

    # Perform the grid search
    grid = GridSearchCV(pipeline, param_grid, cv=sss,
                        n_jobs=-1, scoring="roc_auc", verbose=1)
    encoded_labels = fit_le.transform(train[y_col_name])
    grid.fit(train.drop(y_col_name, axis=1), encoded_labels)
    # Print the results
    print('Best score:', grid.best_score_)
    print('Best parameters:', grid.best_params_)

    return grid


def evaluate_model(best_pipeline: Pipeline, fit_le: LabelEncoder, test: pd.DataFrame, y_col_name:str) -> None:
    clf = best_pipeline["logistic"]

    test_predictions = best_pipeline.predict(
        test.drop(y_col_name, axis=1))
    test_predictions_proba = best_pipeline.predict_proba(
        test.drop(y_col_name, axis=1))

    test_y_encoded = fit_le.transform(test[y_col_name])
    cm = confusion_matrix(
        test_y_encoded, test_predictions, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    sn.heatmap(cm, annot=True, fmt="d", xticklabels=fit_le.classes_,
               yticklabels=fit_le.classes_)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

    # only get predictions from the positive class (=churn)
    PrecisionRecallDisplay.from_predictions(
        test_y_encoded, test_predictions_proba[:, 1], pos_label=1)
    plt.show()