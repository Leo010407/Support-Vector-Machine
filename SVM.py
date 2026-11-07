"""Train a linear SVM and draw the line separating two groups of points.

Run: python svm.py
Save without opening a window: python svm.py --no-show
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

# 1. Create example data. X contains [x1, x2] coordinates;
#    y contains the correct group (0 or 1) for each point.
X, y = make_blobs(
    n_samples=160,
    centers=[(-2, -1), (2, 1)],
    cluster_std=0.9,
    random_state=42,
)

# 2. Reserve 25% of the points for checking predictions after training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 3. Fit the scaler on training data and train a straight-line classifier.
#    Larger C penalizes margin violations more strongly.
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
model.fit(X_train, y_train)

# 4. Evaluate on points the model did not see during training.
print(f"Test accuracy: {model.score(X_test, y_test):.1%}")
new_point = [[1.5, 1.0]]
print(f"Predicted group for {new_point[0]}: {model.predict(new_point)[0]}")

# 5. Ask the trained model for its score across the plane.
#    Negative score = group 0; positive score = group 1; zero = boundary.
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
xx, yy = np.meshgrid(x1, x2)
grid = np.column_stack([xx.ravel(), yy.ravel()])
scores = model.decision_function(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, (scores > 0).astype(int),
            levels=[-0.5, 0.5, 1.5], colors=["#dbeafe", "#ffedd5"], alpha=0.8)
ax.contour(xx, yy, scores, levels=[0], colors="black", linewidths=2)
for group, color in [(0, "#2563eb"), (1, "#ea580c")]:
    points = X_train[y_train == group]
    ax.scatter(points[:, 0], points[:, 1], c=color, edgecolors="white",
               label=f"Group {group} (training)")
    points = X_test[y_test == group]
    ax.scatter(points[:, 0], points[:, 1], c=color, marker="X", s=85,
               edgecolors="black", label=f"Group {group} (test)")

ax.set(xlabel="x1", ylabel="x2", title="Linear SVM: learned dividing line")
ax.legend()
fig.tight_layout()
output = Path(__file__).with_name("svm_example.png")
fig.savefig(output, dpi=160)
print(f"Plot saved to: {output}")
if not args.no_show:
    plt.show()
plt.close(fig)
