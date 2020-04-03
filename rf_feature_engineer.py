from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
from db_query import build_feature_data


def check_impurity(sample_count=1000, n_estimators=100, max_depth=10):
    X, Y, names = build_feature_data(sample_count=sample_count)
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               min_samples_split=0.05,
                               max_features=0.8,
                               max_depth=max_depth,
                               random_state=0)
    rf.fit(X, Y)
    results = round_and_sort(names, rf.feature_importances_)
    return results


def check_acc(sample_count=1000, n_estimators=100, max_depth=10):
    X, Y, names = build_feature_data(sample_count=sample_count)
    rf = RandomForestRegressor(n_estimators=100,
                               min_samples_split=0.05,
                               max_features=0.8,
                               max_depth=10,
                               random_state=0)

    # cv scores with random data perturbations
    rs = ShuffleSplit(n_splits=10,
                      random_state=0,
                      test_size=0.25)
    scores = []
    for train_idx, test_idx in rs.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores.append(abs((acc - shuff_acc)/acc))

    results = round_and_sort(names, scores)
    return results


def round_and_sort(keys, vals, digits=4, reverse=True):
    outs = map(lambda x: round(x, digits), vals)
    return sorted(zip(keys, outs),
                  key=lambda x: x[1],
                  reverse=reverse)
