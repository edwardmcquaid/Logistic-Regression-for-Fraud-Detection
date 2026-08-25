from Data.data_processing import read_and_prepare_data
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    TEST_SIZE,
    TARGET,
    RANDOM_STATE
)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def preprocess_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    #   Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


    #   Now Scale
    scaler = StandardScaler()
    scale_cols = ["Time", "Amount"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.fit_transform(X_test[scale_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test


def fit_regression(xtrain, ytrain, xtest):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(xtrain, ytrain)

    y_pred = model.predict(xtest)
    y_probability = model.predict_proba(xtest)[:, 1]

    return model, y_pred, y_probability


def evaluate(ytest, ypred):
    confusion = confusion_matrix(ytest, ypred)
    return confusion


def main():
    df = read_and_prepare_data(INPUT_DIR / "creditcard.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

    model, y_pred, y_probability = fit_regression(
        X_train_scaled, y_train, X_test_scaled
    )

    print(f"y_pred: {y_pred}")
    print("================")
    print(f"y_prob: {y_probability}")

    confusion = evaluate(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)

    disp.plot()
    plt.show()



if __name__ == "__main__":
    main()
