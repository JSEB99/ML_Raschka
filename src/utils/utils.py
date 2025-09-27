import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_boundary(
        X: np.array,
        y: np.array,
        classifier,
        test_idx: bool = None,
        resolution: float = 0.02,
        x_label: str = 'x',
        y_label: str = 'y',
        legend: bool = True,
        legend_pos: str = 'upper left'
) -> None:
    markers = ('o', 's', '^', 'v', '<')
    colors = ('crimson', 'green', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f'Class {cl}',
            edgecolor='black'
        )

    # Highlight test examples
    if test_idx:
        # plot all examples
        X_test = X[test_idx, :]
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c='none',
            edgecolor='black',
            alpha=1,
            linewidth=1,
            marker='o',
            s=100,
            label='Test set'
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend(loc=legend_pos)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    array1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    array2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*2

    features = np.c_[array1, array2]
    print(features)

    x1, x2 = plot_decision_boundary(features, 1, 1)
    print(x1)
    print(x2)
