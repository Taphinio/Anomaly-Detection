from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X_train, X_test, y_train, y_test = generate_data(
    n_train=300,
    n_test=200,
    n_features=3,
    contamination=0.15,
    random_state=42
)

model_lin = OCSVM(kernel='linear', contamination=0.15)
model_lin.fit(X_train)

y_pred = model_lin.predict(X_test)
scores = model_lin.decision_function(X_test)

ba = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, scores)

print("Linear OCSVM → BA:", ba)
print("Linear OCSVM → AUC:", auc)

fig = plt.figure(figsize=(14,10))

for i,(data,labels,title) in enumerate([
    (X_train,y_train,"Train – Ground truth"),
    (X_test, y_test,"Test – Ground truth"),
    (X_train, model_lin.labels_,"Train – Predicted"),
    (X_test, y_pred,"Test – Predicted"),
]):
    ax = fig.add_subplot(2,2,i+1, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='coolwarm')
    ax.set_title(title)

plt.savefig("linear.png")

model_rbf = OCSVM(kernel='rbf', contamination=0.15)
model_rbf.fit(X_train)

y_pred = model_rbf.predict(X_test)
scores = model_rbf.decision_function(X_test)

ba = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, scores)

print("RBF OCSVM → BA:", ba)
print("RBF OCSVM → AUC:", auc)

fig = plt.figure(figsize=(14,10))

for i,(data,labels,title) in enumerate([
    (X_train,y_train,"Train – Ground truth"),
    (X_test, y_test,"Test – Ground truth"),
    (X_train, model_lin.labels_,"Train – Predicted"),
    (X_test, y_pred,"Test – Predicted"),
]):
    ax = fig.add_subplot(2,2,i+1, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='coolwarm')
    ax.set_title(title)
plt.savefig("rbf.png")

deep = DeepSVDD(n_features=3, contamination=0.15, epochs=30)
deep.fit(X_train)

y_pred = deep.predict(X_test)
scores = deep.decision_function(X_test)

print("DeepSVDD – BA:", balanced_accuracy_score(y_test, y_pred))
print("DeepSVDD – AUC:", roc_auc_score(y_test, scores))

fig = plt.figure(figsize=(14,10))

for i,(data,labels,title) in enumerate([
    (X_train,y_train,"Train – Ground truth"),
    (X_test, y_test,"Test – Ground truth"),
    (X_train, model_lin.labels_,"Train – Predicted"),
    (X_test, y_pred,"Test – Predicted"),
]):
    ax = fig.add_subplot(2,2,i+1, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='coolwarm')
    ax.set_title(title)
plt.savefig("DeepSVDD.png")