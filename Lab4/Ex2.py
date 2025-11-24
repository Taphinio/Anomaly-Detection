from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()
y = 1 - 2*y

X_train, X_test, y_train, y_test  = train_test_split(
    X, y,
    train_size = 0.40,
    random_state=42,
    shuffle=True
)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ocsvm", OneClassSVM())
])

param_grid = {
    "ocsvm__kernel": ["linear", "rbf", "sigmoid"],
    "ocsvm__nu": [0.05, 0.1, 0.15, 0.2, 0.3],
    "ocsvm__gamma": ["scale", "auto", 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n============================")
print("BEST PARAMS FOUND:")
print(grid.best_params_)
print("============================\n")


best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

ba = balanced_accuracy_score(y_test, y_pred)

print("Balanced Accuracy (Test Set):", ba)