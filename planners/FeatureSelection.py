from __future__ import print_function, division

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def rfe_select(tbl):
    """
    Sort features based on recursive feature elemintaion
    """
    clf = LogisticRegression()
    rfe = RFE(clf, n_features_to_select=15)
    numerical_columns = [col for col in tbl.columns if col not in ["F21", "F20", "F54", "Name"]]
    features = tbl[numerical_columns[:-1]]
    klass = tbl[tbl.columns[-1]]
    rfe = rfe.fit(features, klass)
    selected_features = [feat for selected, feat in zip(rfe.support_, numerical_columns) if selected]
    selected_features.insert(0, "Name")
    selected_features.append("category")
    new_tbl = tbl[selected_features]
    # new_tbl.columns = ["Name"] + selected_features[1:]
    return new_tbl
