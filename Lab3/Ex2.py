import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA


X_train, Y = make_blobs(n_samples=1000, n_features=2, 
                        centers=[[10, 0], [0, 10]], 
                        cluster_std=1.0, random_state=42)

X_test = np.random.uniform(-10, 20, (1000, 2))

# IForest
iforest = IForest(contamination=0.02, random_state=42)
iforest.fit(X_train)
scores_iforest = iforest.decision_function(X_test)

# DIF 
dif = DIF(contamination=0.02, hidden_neurons=[32, 16], random_state=42)
dif.fit(X_train)
scores_dif = dif.decision_function(X_test)

# LODA 
loda = LODA(contamination=0.02, n_bins=20)
loda.fit(X_train)
scores_loda = loda.decision_function(X_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# IForest
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=scores_iforest, cmap='viridis', 
                s=20, alpha=0.7)
axes[0].set_title('IForest - artefacte hiperplane paralele cu axele')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# DIF
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=scores_dif, cmap='viridis', 
                s=20, alpha=0.7)
axes[1].set_title('DIF (32,16 neuroni)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

# LODA
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=scores_loda, cmap='viridis', 
                s=20, alpha=0.7)
axes[2].set_title('LODA (20 bins)')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')

for ax in axes:
    plt.colorbar(ax.collections[0], ax=ax, label='Anomaly Score')

plt.tight_layout()
plt.savefig("Ex2.png")
plt.show()

print("\n--- Testare parametri diferiți ---")
# DIF cu mai multe straturi
dif_deep = DIF(contamination=0.02, hidden_neurons=[64, 32, 16], random_state=42)
dif_deep.fit(X_train)
print(f"DIF scor mediu (arhitectură mai adâncă): {dif_deep.decision_function(X_test).mean():.4f}")

# LODA cu bins diferite
loda_10 = LODA(contamination=0.02, n_bins=10)
loda_10.fit(X_train)
print(f"LODA scor mediu (10 bins): {loda_10.decision_function(X_test).mean():.4f}")

loda_30 = LODA(contamination=0.02, n_bins=30)
loda_30.fit(X_train)
print(f"LODA scor mediu (30 bins): {loda_30.decision_function(X_test).mean():.4f}")

print("\n--- Generare versiune 3D ---")
from mpl_toolkits.mplot3d import Axes3D

# Date 3D
X_train_3d, _ = make_blobs(n_samples=1000, n_features=3, 
                           centers=[[0, 10, 0], [10, 0, 10]], 
                           cluster_std=1.0, random_state=42)
X_test_3d = np.random.uniform(-10, 20, (1000, 3))

iforest_3d = IForest(contamination=0.02, random_state=42)
iforest_3d.fit(X_train_3d)
scores_iforest_3d = iforest_3d.decision_function(X_test_3d)

dif_3d = DIF(contamination=0.02, hidden_neurons=[32, 16], random_state=42)
dif_3d.fit(X_train_3d)
scores_dif_3d = dif_3d.decision_function(X_test_3d)

loda_3d = LODA(contamination=0.02, n_bins=20)
loda_3d.fit(X_train_3d)
scores_loda_3d = loda_3d.decision_function(X_test_3d)

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], 
            c=scores_iforest_3d, cmap='viridis', s=20, alpha=0.7)
ax1.set_title('IForest 3D')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], 
            c=scores_dif_3d, cmap='viridis', s=20, alpha=0.7)
ax2.set_title('DIF 3D')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], 
            c=scores_loda_3d, cmap='viridis', s=20, alpha=0.7)
ax3.set_title('LODA 3D')

plt.tight_layout()
plt.savefig("Ex2_3D.png")
plt.show()
