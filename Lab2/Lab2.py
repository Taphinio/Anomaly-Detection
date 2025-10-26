import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pyod.utils.data import generate_data_clusters, make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization


rng = np.random.default_rng()
#Ex 1
def hat_matrix(X):
    XtX = X.T @ X
    L = np.linalg.cholesky(XtX)
    B = np.linalg.solve(L, X.T)      
    A = np.linalg.solve(L.T, B)      
    H = X @ A                        
    return np.diag(H)                

def dataset_1D(n_each=50, a=2.0, b=1.0, mu=0.0, sigma=0.5,
               sigma_high=3.0, x_std_reg=3.0, x_std_high=5.0):
    groups = []

    x_r = rng.normal(0.0, x_std_reg, n_each)
    eps_r = rng.normal(mu, sigma, n_each)
    y_r = a * x_r + b + eps_r
    groups.append(("regular", x_r, y_r))

    x_hx = rng.normal(0.0, x_std_high, n_each)
    eps_hx = rng.normal(mu, sigma, n_each)
    y_hx = a * x_hx + b + eps_hx
    groups.append(("high_var_x", x_hx, y_hx))

    x_hy = rng.normal(0.0, x_std_reg, n_each)
    eps_hy = rng.normal(mu, sigma_high, n_each)  
    y_hy = a * x_hy + b + eps_hy
    groups.append(("high_var_y", x_hy, y_hy))

    x_b = rng.normal(0.0, x_std_high, n_each)
    eps_b = rng.normal(mu, sigma_high, n_each)
    y_b = a * x_b + b + eps_b
    groups.append(("high_var_both", x_b, y_b))

    labels = []
    xs, ys = [], []
    for (name, xg, yg) in groups:
        xs.append(xg)
        ys.append(yg)
        labels += [name] * len(xg) 

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    return X, Y, np.array(labels)

def dataset_2D(n_each = 50, a=1.2, b=0.7, c=0.5, mu=0.0, sigma=0.5, sigma_high=3.0, x_std_reg=1.0,x_std_high = 4.0 ):
    groups = []
    x1_r = rng.normal(0.0, x_std_reg, n_each)
    x2_r = rng.normal(0.0, x_std_reg, n_each)
    eps_r = rng.normal(mu, sigma, n_each)
    y_r = a*x1_r + b*x2_r + c + eps_r
    groups.append(("regular", x1_r, x2_r, y_r))

    x1_hx = rng.normal(0.0, x_std_high, n_each)
    x2_hx = rng.normal(0.0, x_std_high, n_each)
    eps_hx = rng.normal(mu, sigma, n_each)
    y_hx = a*x1_hx + b*x2_hx + c + eps_hx
    groups.append(("high_var_x", x1_hx, x2_hx, y_hx))

    x1_hy = rng.normal(0.0, x_std_reg, n_each)
    x2_hy = rng.normal(0.0, x_std_reg, n_each)
    eps_hy = rng.normal(mu, sigma_high, n_each)
    y_hy = a*x1_hy + b*x2_hy + c + eps_hy
    groups.append(("high_var_y", x1_hy, x2_hy, y_hy))

    x1_b = rng.normal(0.0, x_std_high, n_each)
    x2_b = rng.normal(0.0, x_std_high, n_each)
    eps_b = rng.normal(mu, sigma_high, n_each)
    y_b = a*x1_b + b*x2_b + c + eps_b
    groups.append(("high_var_both", x1_b, x2_b, y_b))
    labels = []
    x1s, x2s, ys = [], [], []
    for (name, x1g, x2g, yg) in groups:
        x1s.append(x1g); x2s.append(x2g); ys.append(yg)
        labels += [name]*len(x1g)
    X1 = np.concatenate(x1s)
    X2 = np.concatenate(x2s)
    Y = np.concatenate(ys)
    return X1, X2, Y, np.array(labels)

def plot1D(mus, sigmas, n_eahc=60, a=2.0, b=1.0):
    for mu in mus:
        for sigma in sigmas:
            X, Y, labels = dataset_1D(n_each=n_eahc, a=a, b=b, mu=mu, sigma=sigma)
            X_design = np.column_stack([np.ones_like(X), X])
            lev = hat_matrix(X_design)

            k = max(5, int(0.03 * len(lev)))
            top_idx = np.argsort(lev)[-k:]

            plt.figure(figsize=(7, 5))
            for grp in np.unique(labels):
                mask = labels == grp        
                plt.scatter(X[mask], Y[mask], label=grp, alpha=0.8)

            plt.scatter(X[top_idx], Y[top_idx], facecolors='none', edgecolors='k',
                        s=120, linewidths=1.5, label='top leverage')

            xs_line = np.linspace(X.min() - 1, X.max() + 1, 200)
            plt.plot(xs_line, a * xs_line + b + mu, linestyle='--', linewidth=1.0)

            plt.title(f"1D linear model: mu={mu}, sigma={sigma}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(f"ex1_1d_mu{mu}_sigma{sigma}.png", dpi=150)
            plt.close()

def plot2D(mus, sigmas, n_each=60, a=1.2, b=0.7, c=0.5):
    for mu in mus:
        for sigma in sigmas:
            X1, X2, Y, labels = dataset_2D(n_each=n_each, a=a, b=b, c=c, mu=mu, sigma=sigma)
            X_design = np.column_stack([np.ones_like(X1), X1, X2])
            lev = hat_matrix(X_design)

            k = max(5, int(0.03 * len(lev)))
            top_idx = np.argsort(lev)[-k:]

            plt.figure(figsize=(7, 5))
            for grp in np.unique(labels):
                mask = labels == grp
                plt.scatter(X1[mask], X2[mask], label=grp, alpha=0.85)

            plt.scatter(X1[top_idx], X2[top_idx], facecolors='none', edgecolors='k',
                        s=120, linewidths=1.5, label='top leverage')

            plt.title(f"2D linear model (Cholesky): mu={mu}, sigma={sigma}")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(f"ex1_2d_mu{mu}_sigma{sigma}_chol.png", dpi=150)
            plt.close()

mus = [0.0, 2.0]
sigmas = [0.2, 1.0, 3.0]

plot1D(mus, sigmas)
plot2D(mus, sigmas)

#Ex 2

def knn_anomaly_detection():
    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=400,
        n_test=200,
        n_clusters=2,
        n_features=2,
        contamination=0.1,
        size='same',
        density='same',
        random_state=42
    )

    n_neighbors_values = [3, 5, 10, 20]

    for n_neighbors in n_neighbors_values:
        clf = KNN(n_neighbors=n_neighbors)
        clf.fit(X_train)

        y_train_pred = clf.labels_
        y_test_pred = clf.predict(X_test)

        ba_train = BA(y_train, y_train_pred)
        ba_test = BA(y_test, y_test_pred)

        print(f"n_neighbors = {n_neighbors} | "
              f"BA train = {ba_train:.3f} | BA test = {ba_test:.3f}")

        plt.figure(figsize=(10, 8))
        plt.suptitle(f"KNN anomaly detection (n_neighbors={n_neighbors})", fontsize=13)

        plt.subplot(2, 2, 1)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=15)
        plt.title("Ground Truth - Train")

        plt.subplot(2, 2, 2)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm', s=15)
        plt.title("Predicted - Train")

        plt.subplot(2, 2, 3)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=15)
        plt.title("Ground Truth - Test")

        plt.subplot(2, 2, 4)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm', s=15)
        plt.title("Predicted - Test")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"ex2_knn_neighbors_{n_neighbors}.png", dpi=150)
        plt.close()

knn_anomaly_detection()

#Ex 3

def ex3_knn_lof(n_neighbors_list=(3, 5, 10, 20), contamination=0.07, random_state=42):
    X, y = make_blobs(
        n_samples=[200, 100],
        centers=[(-10, -10), (10, 10)],
        cluster_std=[2, 6],
        n_features=2,
        shuffle=True,
        random_state=random_state
    )

    for n_neighbors in n_neighbors_list:
        knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
        lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
        knn.fit(X)
        lof.fit(X)

        y_pred_knn = knn.labels_      
        y_pred_lof = lof.labels_

        plt.figure(figsize=(10, 4.5))
        plt.suptitle(f"Ex3 — n_neighbors={n_neighbors}, contamination={contamination}", fontsize=12)

        plt.subplot(1, 2, 1)
        plt.scatter(X[y_pred_knn == 0, 0], X[y_pred_knn == 0, 1], s=15, label="inliers")
        plt.scatter(X[y_pred_knn == 1, 0], X[y_pred_knn == 1, 1], s=20, marker="x", label="outliers")
        plt.title("KNN")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(loc="best")

        plt.subplot(1, 2, 2)
        plt.scatter(X[y_pred_lof == 0, 0], X[y_pred_lof == 0, 1], s=15, label="inliers")
        plt.scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1], s=20, marker="x", label="outliers")
        plt.title("LOF")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"ex3_knn_lof_k{n_neighbors}.png", dpi=150)
        plt.close()

        n_out_knn = int((y_pred_knn == 1).sum())
        n_out_lof = int((y_pred_lof == 1).sum())
        print(f"k={n_neighbors}: KNN outliers={n_out_knn}, LOF outliers={n_out_lof}")

ex3_knn_lof()

#Ex 4

MAT = "cardio.mat"   
MODEL = "KNN"         
KLIST = list(range(30, 121, 10))  
TEST = 0.3
SEED = 42

md = loadmat(MAT)
X, y = md["X"].astype(float), md["y"].ravel().astype(int)
contam = float((y == 1).mean())

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST, stratify=y, random_state=SEED)
Xtr = StandardScaler().fit_transform(Xtr); Xte = StandardScaler().fit(Xtr).transform(Xte) 

Model = KNN if MODEL == "KNN" else LOF
clfs = [Model(n_neighbors=k) for k in KLIST]
for c in clfs: c.fit(Xtr)

ytr_pred = [c.labels_ for c in clfs]
yte_pred = [c.predict(Xte) for c in clfs]
print("Per-model BA:")
for k, yp_tr, yp_te in zip(KLIST, ytr_pred, yte_pred):
    print(f"k={k:3d}  BA_train={BA(ytr, yp_tr):.3f}  BA_test={BA(yte, yp_te):.3f}")

Str = np.column_stack([c.decision_scores_ for c in clfs])
Ste = np.column_stack([c.decision_function(Xte) for c in clfs])
Str_n, Ste_n = standardizer(Str, Ste)

avg_tr, avg_te = average(Str_n), average(Ste_n)
max_tr, max_te = maximization(Str_n), maximization(Ste_n)

thr_avg = np.quantile(avg_tr, 1.0 - contam)
thr_max = np.quantile(max_tr, 1.0 - contam)

ytr_avg = (avg_tr >= thr_avg).astype(int); yte_avg = (avg_te >= thr_avg).astype(int)
ytr_max = (max_tr >= thr_max).astype(int); yte_max = (max_te >= thr_max).astype(int)

print("\nEnsemble (normalized scores):")
print(f"AVERAGE  BA_train={BA(ytr, ytr_avg):.3f}  BA_test={BA(yte, yte_avg):.3f}")
print(f"MAX      BA_train={BA(ytr, ytr_max):.3f}  BA_test={BA(yte, yte_max):.3f}")