import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_blobs(n_samples=160, centers=[(-2, -1), (2, 1)],
                  cluster_std=0.9, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1))
clf.fit(X_train, y_train)

print("test accuracy:", clf.score(X_test, y_test))
print("prediction for (1.5, 1.0):", clf.predict([[1.5, 1.0]])[0])

# grid of points to draw the boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2)
plt.contour(xx, yy, Z, levels=[0], colors="k")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label="train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", s=60, label="test")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("linear SVM")
plt.legend()
plt.savefig("svm.png", dpi=150)
plt.show()
