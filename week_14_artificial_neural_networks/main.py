import pandas as pd
import sklearn.neural_network as nn
from sklearn.decomposition import PCA
import sklearn.model_selection as ms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


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

    # build a model
    mlp_model = nn.MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        hidden_layer_sizes=(100, 100)
    )
    mlp_model.fit(x_r, y)

    # Calculate the model score when evaluated using 5-cross-validation
    results = ms.cross_val_score(mlp_model, x_r, y, cv=5)
    print("Model Score=", results.mean())

    # A sample prediction (row n)
    n = 101  # a sample row number
    z = x_r[n:n + 1, :]
    prediction_proba = mlp_model.predict_proba(z)
    prediction_value = prediction_proba[0][1]
    prediction_class = 'M' if prediction_value.round() == 1 else 'B'
    actual_class = 'M' if y[n].round() == 1 else 'B'
    print("The prediction for sample #", n, " is:", prediction_value, "(", prediction_class, ")",
          "Actual diagnosis was:", actual_class)

    # # print(f'weight value coefficient: {mlp_model.coefs_}')
    # for idx, weight_matrix in enumerate(mlp_model.coefs_):
    #     print(f'Shape of weight matrix {idx}: {weight_matrix.shape}')
    #     # Optional: Visualize the weight matrix
    #     plt.figure(figsize=(10, 5))
    #     sns.heatmap(weight_matrix, cmap='viridis')
    #     plt.title(f'Heatmap of weight matrix {idx}')
    #     plt.show()

    # plt.plot(mlp_model.loss_curve_)
    # plt.show()


if __name__ == "__main__":
    main()
