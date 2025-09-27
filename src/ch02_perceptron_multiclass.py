import numpy as np
import pandas as pd
from ch02_perceptron import Perceptron
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class OvRPerceptron():
    """
    Implements a Perceptron One vs Rest or One vs All
    for multiclass tasks

    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.

    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    b_ : Scalar
    Bias unit after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta  # Learning Rate
        self.n_iter = n_iter  # Epochs
        self.random_state = random_state  # Manage randomness

    def ovr_method(self, X: np.array, y: np.array) -> "OvRPerceptron":
        """
        One VS Rest method through Perceptron Class

        :param X: attributes matrix, shape=[n_samples, n_features]
        :type X: numpy array matrix 
        :param y: targets vector, shape=[n_samples]
        :type y: numpy array vector

        :return: classifier
        :rtype: object with a classifiers dict[weights, bias, errors], classes array
        """
        self.classifiers_: dict[str, Perceptron] = {}
        self.classes_: np.array = np.unique(y)
        targets: np.array = np.array([
            np.where(y == target, 1, 0)
            for target in self.classes_])

        for idx, cls in enumerate(self.classes_):
            clf = Perceptron(
                eta=self.eta,
                n_iter=self.n_iter,
                random_state=self.random_state)
            clf.fit(X, targets[idx])
            self.classifiers_[cls] = clf
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict over an array of features

        :param X: 
        :param X: attributes matrix, shape=[n_samples, n_features]
        :type X: numpy array matrix 

        :return: predictions
        :rtype: numpy array
        """
        scores = {}
        for cls, clf in self.classifiers_.items():
            scores[cls] = clf.net_input(X)

        all_scores = np.column_stack([scores[cls] for cls in self.classes_])
        predictions = self.classes_[np.argmax(all_scores, axis=1)]

        return predictions

    def plot_errors(self, title='Errores por Época (One-vs-Rest)'):
        plt.figure(figsize=(8, 5))

        for cls, clf in self.classifiers.items():
            plt.plot(
                range(1, len(clf.errors_) + 1), clf.errors_,
                marker='o', label=f'Clase {cls}')

        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Número de Updates')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_decision_regions(X, y, classifier, resolution=0.02):
        # Definir marcadores y colores para varias clases
        markers = (
            'o', 's', '^', 'v', '<', '>', 'x', 'D',
            'p', '*')  # ampliado por si hay más clase
        colors = (
            'red', 'blue', 'lightgreen', 'gray', 'cyan',
            'purple', 'orange', 'brown', 'pink', 'olive')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Limites del plot
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Crear la malla de puntos
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution))

        # Predecir para todos los puntos de la malla
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        # Graficar la superficie de decisión
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # Graficar muestras de entrenamiento por clase
        for idx, label in enumerate(np.unique(y)):
            plt.scatter(x=X[y == label, 0],
                        y=X[y == label, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=f'Clase {label}',
                        edgecolor='black',
                        s=60)

        plt.legend(loc='upper left')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Regiones de decisión')
        plt.show()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=3,
        class_sep=2.5,  # aumenta la separabilidad
        random_state=1)

    ovr = OvRPerceptron(eta=0.1, n_iter=10)
    ovr.ovr_method(X, y)
    # print(y)
    y_preds = ovr.predict(X)
    # print(y_preds)
    # print(y_preds == y)
    resultados = {}
    for cls in range(3):
        print(f'Test {cls+1}: X = {X[cls+1]}, y = {y[cls+1]}')
    for cls, clf in ovr.classifiers_.items():
        print(f"\n{'Test '+str(cls):_^70}\n")
        print(f'Clase {cls} pesos: {clf.w_} y biases: {clf.b_}')
        # ✅ Predicciones para cada muestra por clase
        # La mas alta entre las muestras es la clase, es decir si la mas alta
        # fue la primera entonces es clase 0, es decir, predice 0
        # Si el target de esa muestra es 0, entonces predijo CORRECTAMENTE
        print('Clase 0:', X[3], y[3], clf.net_input(X[3]))
        print('Clase 1:', X[1], y[1], clf.net_input(X[1]))
        print('Clase 2:', X[2], y[2], clf.net_input(X[2]))
        # ✅ Añado cada predicción de cada muestra a su supuesto predictor en ORDEN
        resultados.setdefault('Clase 0', []).append(clf.net_input(X[3]))
        resultados.setdefault('Clase 1', []).append(clf.net_input(X[1]))
        resultados.setdefault('Clase 2', []).append(clf.net_input(X[2]))

    print(f"\n{'Resultados':_^70}\n")
    # ✅ Por cada clase escojo el mayor, al ser añadidos en orden el index que retorne corresponde tambien a la clase
    # Si en la clase 0, el índice mas alto fue el 0, predice correctamente y asi sucesivamente
    print(f'{X[3]} es clase {y[3]} y el predictor dijo: {np.array(resultados['Clase 0']).argmax()}')
    print(f'{X[1]} es clase {y[1]} y el predictor dijo: {np.array(resultados['Clase 1']).argmax()}')
    print(f'{X[2]} es clase {y[2]} y el predictor dijo: {np.array(resultados['Clase 2']).argmax()}')
