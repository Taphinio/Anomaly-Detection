import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF


data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

iforest_ba, iforest_auc = [], []
loda_ba, loda_auc = [], []
dif_ba, dif_auc = [], []

print("Rulez 10 split-uri diferite...")
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42+i, stratify=y
    )
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    contamination_rate = np.mean(y_train == 1)  # 1 = anomalie în ODDS
        
    # IForest
    model = IForest(contamination=contamination_rate, random_state=42+i)
    model.fit(X_train_norm)
    iforest_ba.append(balanced_accuracy_score(y_test, model.predict(X_test_norm)))
    iforest_auc.append(roc_auc_score(y_test, model.decision_function(X_test_norm)))
    
    # LODA
    model = LODA(contamination=contamination_rate)
    model.fit(X_train_norm)
    loda_ba.append(balanced_accuracy_score(y_test, model.predict(X_test_norm)))
    loda_auc.append(roc_auc_score(y_test, model.decision_function(X_test_norm)))
    
    # DIF
    model = DIF(contamination=contamination_rate, random_state=42+i)
    model.fit(X_train_norm)
    dif_ba.append(balanced_accuracy_score(y_test, model.predict(X_test_norm)))
    dif_auc.append(roc_auc_score(y_test, model.decision_function(X_test_norm)))
    
print("\n--- Rezultate medii (10 split-uri) ---")
print(f"IForest - BA: {np.mean(iforest_ba):.4f}, ROC AUC: {np.mean(iforest_auc):.4f}")
print(f"LODA    - BA: {np.mean(loda_ba):.4f}, ROC AUC: {np.mean(loda_auc):.4f}")
print(f"DIF     - BA: {np.mean(dif_ba):.4f}, ROC AUC: {np.mean(dif_auc):.4f}")