import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.utils import plot_decision_boundary

# data import
data = load_iris()
X, y = data.data[:, [2, 3]], data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
# standarize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# whole standarize data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# model
svm = SVC(
    C=1.0,
    kernel='linear',
    random_state=1
)
t1 = time.time()
# model training
svm.fit(X_train_std, y_train)
t2 = time.time()
# model evaluation
plot_decision_boundary(
    X_combined_std,
    y_combined,
    svm,
    test_idx=range(105, 150),
    x_label='Petal length [standarized]',
    y_label='Petal width [standarized]'
)
# SGD Alternative version
t3 = time.time()
svm_sgd = SGDClassifier(
    loss='hinge',
    alpha=1/(len(X_train_std)*1),
    random_state=1
)
svm_sgd.fit(X_train_std, y_train)
t4 = time.time()
# model evaluation
plot_decision_boundary(
    X_combined_std,
    y_combined,
    svm_sgd,
    test_idx=range(105, 150),
    x_label='Petal length [standarized]',
    y_label='Petal width [standarized]'
)
print('Duración sin SGD:', round(t2-t1, 4))
print('Duración con SGD:', round(t4-t3, 4))
