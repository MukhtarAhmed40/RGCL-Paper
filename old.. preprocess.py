import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(file):
    df = pd.read_csv(file)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
