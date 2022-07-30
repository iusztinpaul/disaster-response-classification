import argparse
import pickle
import re
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import nltk
import yaml

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine

nltk.download(["punkt", "stopwords", "wordnet", "omw-1.4"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "-database-filepath",
    required=True,
    help="Absolute or relative path to the Sqlite database."
)
parser.add_argument(
    "-model-filepath",
    required=True,
    help="Absolute or relative path to the pickle file to save the model to."
)
parser.add_argument(
    "-config-filepath",
    required=True,
    help="Absolute or relative path to the configuration file."
)

parser.add_argument(
    "-run-gridsearch",
    default=False,
    type=bool,
    help="Weather to run a simple train or a gridsearch."
)
parser.add_argument(
    "-debug",
    default=False,
    type=bool,
    help="Set the running mode to debug."
)

STOP_WORDS = set(stopwords.words("english"))


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads the data from the Sqlite database.

    @param database_filepath: Path to the Sqlite database.
    @return: The loaded features, targets and target names.
    """

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("processed", con=engine)

    X = df[["message", "genre"]]
    Y = df.select_dtypes(include="int").drop(columns=["id"])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text: str) -> List[str]:
    """
    Function that tokenizes a message from the dataset.

    @param text: The raw text to tokenize.
    @return: A list of processed tokens.
    """

    # Normalize.
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize.
    tokens = word_tokenize(text)

    # Clean.
    tokens = [token for token in tokens if token not in STOP_WORDS]

    # Lemmatize.
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def build_model(**kwargs):
    """
    Builds the model pipeline.

    @param kwargs: The hyperparameters for the model.
    @return: The model pipeline.
    """

    message_pipeline = Pipeline([
        ("count", CountVectorizer(tokenizer=tokenize)),
        ("tfid", TfidfTransformer())
    ])
    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
    ])
    features_pipeline = ColumnTransformer(transformers=[
        ("message", message_pipeline, "message"),
        ("categorical", categorical_pipeline, ["genre"])
    ])

    classifier = build_classifier(
        type=kwargs.pop("type"),
        **kwargs
    )
    model = Pipeline([
        ("features", features_pipeline),
        ("classifier", classifier)
    ], memory="./cache", verbose=True)

    return model


def build_classifier(type: str, **kwargs):
    """
    Builds a classifier based on the given type.

    @param type: Possible classifier types: "random_forest", "logistic_regression" or "naive_bayes".
    @param kwargs: The hyperparameters for the classifier.
    @return: A classifier.
    """

    assert type in ("random_forest", "logistic_regression", "naive_bayes")

    if type == "random_forest":
        classifier = RandomForestClassifier(**kwargs.get("hyper_parameters", {}))
    elif type == "logistic_regression":
        classifier = LogisticRegression(**kwargs.get("hyper_parameters", {}))
    else:
        classifier = GaussianNB(**kwargs.get("hyper_parameters", {}))

    return MultiOutputClassifier(classifier)


def build_gridsearch(model, **kwargs):
    """
    Adds gridsearch to the model.

    @param model: The desired to run gridsearch on.
    @return: The best model.
    """

    parameters = kwargs["search_hyper_parameters"]
    cv = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring=make_scorer(multiclass_f1_score)
    )

    return cv


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Evaluates the model on the given test set.

    @param model: The trained model to evaluate.
    @param X_test: The test set features.
    @param y_test: The test set labels.
    @return: The mean over all the targets of the F1 score.
    """

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

    f1_score = multiclass_f1_score(y_test, y_pred)
    print(f"F1 score: {f1_score}")

    return f1_score


def multiclass_f1_score(
        y_true: pd.DataFrame,
        y_pred: Union[pd.DataFrame, np.ndarray],
        verbose: bool = False
) -> float:
    """
    A function that measures the mean of the weighted mean F1 for a multiclass prediction setup.

    @param y_true: A pandas DataFrame with true labels.
    @param y_pred: A pandas DataFrame with predicted labels.
    @param verbose: A boolean value that determines whether to print the results for every target.
    @return: A float value that represents the mean of the weighted F1 for a multiclass prediction setup.
    """

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=y_true.columns)

    assert (y_true.columns == y_pred.columns).all(), "The columns of y_true and y_pred must be the same."

    averages = []
    for target in y_true.columns:
        target_results = classification_report(
            output_dict=True,
            y_true=y_true[target],
            y_pred=y_pred[target],
            labels=y_pred[target].unique(),
        )
        eval_df = pd.DataFrame.from_dict(target_results, orient="columns")

        if verbose is True:
            print(f"Target {target}")
            print(eval_df)

        averages.append(eval_df.loc["f1-score", "weighted avg"])

    return np.array(averages, dtype=np.float64).mean()


def save_model(model, model_filepath: str):
    """
    Saves the model as a pickle to the given filepath.

    @param model: The model to be saved.
    @param model_filepath: Path to the file to save the model to.
    @return: None
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """
    Main function that aggregates all the logic.

    @return: None
    """

    args = parser.parse_args()

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    run_gridsearch = args.run_gridsearch

    print("Loading data...\n    DATABASE: {}".format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open(args.config_filepath) as f:
        hyper_parameters = yaml.safe_load(f)
    print("Config: ")
    print(f"Loaded from {args.config_filepath}")
    print(hyper_parameters)

    print("Building model...")
    model = build_model(**hyper_parameters)

    if run_gridsearch is True:
        print("Building gridsearch...")
        model = build_gridsearch(model)

    print("Training model(s)...")
    model.fit(X_train, Y_train)

    if run_gridsearch is True:
        print("\n The best score across ALL searched params:\n", model.best_score_)
        print("\n The best parameters across ALL searched params:\n", model.best_params_)

        model = model.best_estimator_

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test)

    print("Saving model...\n    MODEL: {}".format(model_filepath))
    save_model(model, model_filepath)

    print("Trained model saved!")


if __name__ == "__main__":
    main()
