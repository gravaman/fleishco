from sklearn.ensemble import RandomForestRegressor
from db_query import build_feature_data


def test_data():
    X, Y, names = build_feature_data(sample_count=1000)
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    results = round_and_sort(names, rf.feature_importances_)
    print(results)


def round_and_sort(keys, vals, digits=4, reverse=True):
    outs = map(lambda x: round(x, digits), vals)
    return sorted(zip(keys, outs), key=lambda x: x[1],
                  reverse=reverse)


test_data()
