from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans


def eval_classification(X, y, p=0.3):
    # print(X.shape)
    # X = X / np.tile(np.array(np.linalg.norm(X, axis=1)), (X.shape[1], 1)).T
    # print(np.linalg.norm(X))
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1.0 - p, random_state=222)
    clf = LinearSVC()
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    # print(set(y_pred))
    macro = f1_score(y_pred, test_y, average='macro')
    micro = f1_score(y_pred, test_y, average='micro')
    accuracy = accuracy_score(test_y, y_pred)
    print('Macro:{:.4f} Micro:{:.4f} Acc:{:.4f}'.format(macro,micro,accuracy))
    return macro, micro, accuracy


def eval_link_prediction(embeddings, test_lp_data):
    (a, b, l) = test_lp_data
    x = embeddings[a]
    y = embeddings[b]
    norm_x = (x.T / np.linalg.norm(x, axis=1)).T
    norm_y = (y.T / np.linalg.norm(y, axis=1)).T
    cos = np.sum(norm_x * norm_y, axis=1)
    auc = roc_auc_score(y_true=l, y_score=cos)
    print('auc is ', auc)


def eval_clustering(predict, label):
    nmi = normalized_mutual_info_score(labels_true=label, labels_pred=predict)
    print('nmi = ', nmi)
    return nmi


def eval_kmeans(feature, label):
    K = len(set(label))
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(feature)
    nmi = normalized_mutual_info_score(labels_true=label, labels_pred=kmeans.labels_)
    acc = get_acc(y_true=label, y_pred=kmeans.labels_)
    print('nmi is %f, acc is %f' % (nmi, acc))


def get_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
