import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "C:/Users/Buse/desktop/PythonProgramlama/veri_manipülasyonu/Calısmalarım/breastCancer.csv"
cancer = pd.read_csv(path)
df = cancer.copy()
print(df.head())
print(df.info())
df.drop(columns=["id", "Unnamed: 32"], axis = 1, inplace=True)
le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["diagnosis"])
df.drop(columns=["diagnosis"], axis=1, inplace=True)
print(df.head())

X = df.drop(["Diagnosis"], axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 3, stratify=y)
print("y_test value counts: ")
print(y_test.value_counts())

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

print("Train score: " , accuracy_score(y_train, y_pred_train))
print("Test score: ", accuracy_score(y_test, y_pred_test))

