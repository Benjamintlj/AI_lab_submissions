import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
from sklearn.decomposition import PCA


def main():
    # Read data
    data = pd.read_csv('wdbc.csv', header=None)
    data = data.replace({'B': 0, 'M': 1})
    x = data.iloc[:, 2:]
    y = data.iloc[:, 1]

    # Scaling
    x = (x - x.mean()) / x.std()

    # PCA
    pca = PCA(n_components=1)
    x_r = pca.fit_transform(x)

    # Build a regression model and fit it with x_r, y
    model = lm.LogisticRegression(max_iter=10000)
    model.fit(x_r, y)

    # Calculate the model score when evaluated using 5-cross-validation
    results = ms.cross_val_score(model, x_r, y, cv=5)
    print("Model Score=", results.mean())

    # A sample prediction (row n)
    n = 101  # a sample row number
    z = x_r[n:n + 1, :]
    prediction_proba = model.predict_proba(z)
    prediction_value = prediction_proba[0][1]
    prediction_class = 'M' if prediction_value.round() == 1 else 'B'
    actual_class = 'M' if y[n].round() == 1 else 'B'
    print("The prediction for sample #", n, " is:", prediction_value, "(", prediction_class, ")",
          "Actual diagnosis was:", actual_class)


if __name__ == "__main__":
    main()
