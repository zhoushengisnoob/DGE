from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

def eval_classification(X, y, p=0.3):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1.0 - p, random_state=6)
    clf = LinearSVC()
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    macro = f1_score(y_pred, test_y, average='macro')
    micro = f1_score(y_pred, test_y, average='micro')
    accuracy = accuracy_score(test_y, y_pred)
    return macro, micro, accuracy