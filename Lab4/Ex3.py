from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
import numpy as np

data = loadmat("shuttle.mat")

X = data["X"]
y = data["y"].ravel()

print("Loaded:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.5,
    random_state = 42,
    shuffle=True
)

X_train_s, X_test_s = standardizer(X_train, X_test)
X_train_s = X_train_s.astype(np.float32)
X_test_s = X_test_s.astype(np.float32)
cont = y_train.mean() 

ocsvm = OCSVM(kernel="rbf", contamination=cont)
ocsvm.fit(X_train_s)

yp_oc = ocsvm.predict(X_test_s)
score_oc = ocsvm.decision_function(X_test_s)

ba_oc = balanced_accuracy_score(y_test, yp_oc)
auc_oc = roc_auc_score(y_test, score_oc)

print("\n===============================")
print("OCSVM Results:")
print("Balanced Accuracy:", ba_oc)
print("ROC AUC:", auc_oc)
print("===============================\n")

architectures = [
    [64, 32],
    [128, 64],
    [128, 64, 32]
]
print("DeepSVDD Results:\n")

for arch in architectures:
    print(f"Architecture: {arch}")

    deep = DeepSVDD(
        n_features=X_train_s.shape[1],
        hidden_neurons=arch,
        epochs=40,
        contamination=cont
    )

    deep.fit(X_train_s)

    yp = deep.predict(X_test_s)
    score = deep.decision_function(X_test_s)

    ba = balanced_accuracy_score(y_test, yp)
    auc = roc_auc_score(y_test, score)

    print("  Balanced Accuracy:", ba)
    print("  ROC AUC:", auc)
    print("----------------------------------")