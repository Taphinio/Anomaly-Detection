from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix , balanced_accuracy_score , roc_curve, auc
import numpy as np

#Ex 1
x_train, x_test ,y_train, y_test = generate_data(
        n_train = 400,
        n_test = 100,
        n_features = 2,
        contamination = 0.1,
        random_state = 42
)

plt.figure(figsize=(6,6))

plt.scatter(x_train[y_train == 0 , 0] , x_train[y_train == 0 , 1] ,
                c ='blue' , label = 'Normal' , alpha = 0.6 )

plt.scatter(x_train[y_train == 1 , 0] , x_train[y_train == 1 , 1] ,
                c ='red' , label = 'Outlier' , alpha = 0.6 )

plt.title("Monstre de antrenare: Normal vs Outlier")
plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
plt.legend()
plt.savefig("ex1.svg")
plt.show()


# #Ex 2

def evaluate_knn( model_contamination : float):

    clf = KNN( contamination = model_contamination)
    clf.fit(x_train)

    y_pred_train = clf.labels_
    y_pred_test = clf.predict(x_test)

    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_pred_train).ravel()
    bacc_tr = balanced_accuracy_score(y_train, y_pred_train)

    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_pred_test).ravel()
    bacc_te = balanced_accuracy_score(y_test, y_pred_test)

    test_scores = clf.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize = (6,6))
    plt.plot(fpr, tpr, label = f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1] , [0,1], linestyle = '--', label = 'Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'KNN ROC on Test ( model contamination = {model_contamination})')
    plt.legend()
    plt.tight_layout()
    outfile = f'roc_knn_contam_{str(model_contamination).replace(".", "")}.png'
    plt.savefig(outfile,)

    print(f'\n=== Model contamination = {model_contamination} ===')
    print(f'[TRAIN] TN, FP, FN, TP = {tn_tr}, {fp_tr}, {fn_tr}, {tp_tr} | Balanced Acc = {bacc_tr:.3f}')
    print(f'[TEST ] TN, FP, FN, TP = {tn_te}, {fp_te}, {fn_te}, {tp_te} | Balanced Acc = {bacc_te:.3f}')
    print(f'[TEST ] AUC (ROC) = {roc_auc:.3f} | ROC image saved to: {outfile}')
    return {
        "contamination": model_contamination,
        "train": {"tn": tn_tr, "fp": fp_tr, "fn": fn_tr, "tp": tp_tr, "bacc": bacc_tr},
        "test":  {"tn": tn_te, "fp": fp_te, "fn": fn_te, "tp": tp_te, "bacc": bacc_tr, "auc": roc_auc},
    }

results = []
results.append(evaluate_knn(0.1))

for c in [0.02, 0.05, 0.2, 0.3]:
    results.append(evaluate_knn(c))

print("\nSummary (test balanced accuracy vs model contamination):")
for r in results:
    print(f"  c={r['contamination']:<4}: bAcc={r['test']['bacc']:.3f}, AUC={r['test']['auc']:.3f}")

#Ex 3

contam = 0.1

x_train , x_test, y_train, y_pred= generate_data(

    n_train = 1000,
    n_test = 0,
    n_features = 1,
    contamination = contam, 
    random_state = 42
)
plt.figure(figsize=(6,6))

plt.scatter(x_train[y_train == 0, 0 ] , [0] * len(x_train[y_train == 0, 0 ]),
                c ='blue' , label = 'Normal' , alpha = 0.6 )

plt.scatter(x_train[y_train == 1 , 0] , [0] * len(x_train[y_train == 1, 0 ]),
                c ='red' , label = 'Outlier' , alpha = 0.6 )

plt.title("Monstre de antrenare: Normal vs Outlier")
plt.legend()
plt.show()
x = x_train[:, 0]

mu = np.mean(x)
sigma = np.std(x, ddof=0)
z = np.abs((x-mu) / sigma) if sigma > 0 else np.zeros_like(x)

thr = np.quantile(z, 1 - contam)

y_pred = ( z >= thr).astype(int)

tn , tp , fn, fp = confusion_matrix(y_train, y_pred).ravel()
bacc = balanced_accuracy_score(y_train, y_pred)

print(f"Mean={mu:.4f}, Std={sigma:.4f}")
print(f"Z-threshold (|z|) for {contam*100:.0f}% contamination: {thr:.4f}")
print(f"Predicted contamination: {y_pred.mean():.3f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Balanced accuracy: {bacc:.3f}")

# Ex 4

contam = 0.1

x_train , x_test, y_train, y_pred= generate_data(

    n_train = 1000,
    n_test = 0,
    n_features = 2,
    contamination = contam, 
    random_state = 42
)

mu = np.array([2, -1])
Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
L = np.linalg.cholesky(Sigma)
Y = x_train @ L.T + mu

z = np.linalg.norm((Y - Y.mean(axis=0)) / Y.std(axis=0), axis=1)
th = np.quantile(z, 1 - 0.1)
y_pred = (z > th).astype(int)
bal_acc = balanced_accuracy_score(y_train, y_pred)
print(th, bal_acc)

plt.scatter(Y[y_pred==0,0], Y[y_pred==0,1], s=15, c="green", label="Normal")
plt.scatter(Y[y_pred==1,0], Y[y_pred==1,1], s=15, c="red", label="Anomaly")
plt.title(f"Detected Anomalies (Balanced Acc={bal_acc:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("ex4.svg")
plt.show()