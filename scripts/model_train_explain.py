import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

    
def load_dataset(path):
    df = pd.read_parquet(path)
    return df

def _rating_to_bucket(rating):
    if rating < 6:
        return 0
    else:
        return 1

def preprocess_dataset(df):
    df['averageRatingBucket'] = df['averageRating'].apply(_rating_to_bucket)
    df = df.drop(['primaryTitle'], axis=1)

    return df

def get_splits(df, test_size=0.2):
    X = df.drop(['averageRating', 'averageRatingBucket'], axis=1)
    y = df['averageRatingBucket']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        stratify=df['averageRatingBucket'],
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = xgboost.train(
        {
            "learning_rate": 0.1,
            "max_depth": 5,
            "objective": "binary:logistic",
            "base_score": y_train.mean()
        },
        xgboost.DMatrix(X_train, label=y_train),
        num_boost_round=500
    )
        
    return model

def score_model(model, X_test, y_test):
    dtest = xgboost.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest) > 0.5

    print(accuracy_score(y_test, y_pred))

def explain_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
