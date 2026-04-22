import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
import os

classes = ['http', 'ftp', 'ssh', 'dns', 'p2p']
X_train_list = []
y_train_list = []

for cls in classes:
    filename = f'r_{cls}.txt'
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        X_train_list.append(data)
        y_train_list.extend([cls] * data.shape[0])

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

X_unlabeled = None
if os.path.exists('unmarked.txt'):
    X_unlabeled = np.loadtxt('unmarked.txt')

test_order = ['http', 'ftp', 'ssh', 'dns', 'p2p']
X_test_list = []
for cls in test_order:
    data = np.loadtxt(f'test_{cls}.txt')
    X_test_list.append(data)

X_test = np.vstack(X_test_list)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
if X_unlabeled is not None:
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

base_clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

if X_unlabeled is not None:
    clf = SelfTrainingClassifier(
        base_clf,
        threshold=0.9,
        criterion='threshold',
        max_iter=10,
        verbose=True
    )
    X_combined = np.vstack([X_train_scaled, X_unlabeled_scaled])
    y_combined = np.concatenate([y_train_enc, np.full(X_unlabeled_scaled.shape[0], -1)])
    clf.fit(X_combined, y_combined)
else:
    clf = base_clf
    clf.fit(X_train_scaled, y_train_enc)

y_pred_enc = clf.predict(X_test_scaled)
y_pred = le.inverse_transform(y_pred_enc)

class_mapping = {'http':'HTTP','ftp':'FTP','ssh':'SSH','dns':'DNS','p2p':'P2P'}
y_pred_mapped = [class_mapping[label] for label in y_pred]

with open('predictions.txt', 'w', encoding='utf-8') as f:
    for pred in y_pred_mapped:
        f.write(pred + '\n')

final_estimator = clf.base_estimator_ if hasattr(clf, 'base_estimator_') else clf
print(f"Количество параметров модели: {final_estimator.coef_.size + final_estimator.intercept_.size}")