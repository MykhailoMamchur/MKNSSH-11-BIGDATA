import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

    
def load_dataset(path):
    """
    Loads a dataset from a parquet file.

    Parameters:
    ----------
    path (str): The path to the parquet file.

    Returns:
    ----------
    DataFrame: The loaded dataset as a pandas DataFrame.
    """

    df = pd.read_parquet(path)
    return df

def _rating_to_bucket(rating):
    """
    Converts a rating into a binary bucket (0 for ratings below 6, 1 for ratings 6 and above).

    Parameters:
    ----------
    rating (float): The rating to be converted.

    Returns:
    ----------
    int: 0 if rating is below 6, 1 otherwise.
    """
    if rating < 6:
        return 0
    else:
        return 1

def preprocess_dataset(df):
    """
    Preprocesses the dataset by adding a bucket for ratings and removing the 'primaryTitle' column.

    Parameters:
    ----------
    df (DataFrame): The input dataset.

    Returns:
    ----------
    DataFrame: The processed dataset with the 'averageRatingBucket' column and without the 'primaryTitle' column.
    """

    df['averageRatingBucket'] = df['averageRating'].apply(_rating_to_bucket)
    df = df.drop(['primaryTitle'], axis=1)

    return df

def get_splits(df, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    ----------
    df (DataFrame): The dataset to be split.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    ----------
    tuple: (X_train, X_test, y_train, y_test) where X is the feature set and y is the target.
    """

    X = df.drop(['averageRating', 'averageRatingBucket'], axis=1)
    y = df['averageRatingBucket']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        stratify=df['averageRatingBucket'],
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains an XGBoost model for binary classification.

    Parameters:
    ----------
    X_train (DataFrame): The feature set for training.
    y_train (Series): The target labels for training.

    Returns:
    ----------
    model: The trained XGBoost model.
    """

    model = xgboost.train(
        {
            "learning_rate": 0.1,
            "max_depth": 12,
            "objective": "binary:logistic",
            "base_score": y_train.mean()
        },
        xgboost.DMatrix(X_train, label=y_train),
        num_boost_round=200
    )
        
    return model

def score_model(model, X, y):
    """
    Scores the model using accuracy.

    Parameters:
    ----------
    model: The trained model.
    X (DataFrame): The feature set.
    y (Series): The true labels.

    Returns:
    ----------
    float: The accuracy score of the model.
    """

    dtest = xgboost.DMatrix(X, label=y)
    y_pred = model.predict(dtest) > 0.5

    return round(accuracy_score(y, y_pred), 2)

def score_model_f1(model, X, y):
    """
    Scores the model using the F1 score.

    Parameters:
    model: The trained model.
    X (DataFrame): The feature set.
    y (Series): The true labels.

    Returns:
    float: The F1 score of the model.
    """

    dtest = xgboost.DMatrix(X, label=y)
    y_pred = model.predict(dtest) > 0.5

    return round(f1_score(y, y_pred), 2)

def score_model_show_cm(model, X, y, title=''):
    """
    Plots the confusion matrix for the model's predictions.

    Parameters:
    ----------
    model: The trained model.
    X (DataFrame): The feature set.
    y (Series): The true labels.
    title (str): The title of the confusion matrix plot.
    """

    dtest = xgboost.DMatrix(X, label=y)
    y_pred = model.predict(dtest) > 0.5
    
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    plt.title(title)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)

def explain_model(model, X):
    """
    Explains the model's predictions using SHAP (SHapley Additive exPlanations).

    Parameters:
    ----------
    model: The trained model.
    X (DataFrame): The feature set.
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
