import numpy as np


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return 'aishwary'

    def add_evidence(self, data_x, data_y):
        self.tree = self._build_tree(data_x, data_y)

        if self.verbose:
            print("DTLearner")
            print("Tree Shape: ", self.tree.shape)
            print("Tree Details: \n", self.tree)

    def query(self, points):
        return np.array([self._get_prediction(point) for point in points])

    def _get_prediction(self, point):
        node_index = 0
        while not np.isnan(self.tree[node_index, 0]):
            feature_index, split_value, left_child, right_child = self.tree[node_index, 0:4]
            node_index += int(left_child) if point[int(feature_index)] <= split_value else int(right_child)
        return self.tree[node_index, 1]

    def _build_tree(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size:
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])

        if len(np.unique(data_y)) == 1:
            return np.array([[np.nan, data_y[0], np.nan, np.nan]])

        feature = self._get_best_feature(data_x, data_y)
        split_value = np.median(data_x[:, feature])

        if np.allclose((data_x[:, feature] <= split_value), (data_x[:, feature] <= split_value)[0]):
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])

        left_tree = self._build_tree(data_x[data_x[:, feature] <= split_value], data_y[data_x[:, feature] <= split_value])
        right_tree = self._build_tree(data_x[data_x[:, feature] > split_value], data_y[data_x[:, feature] > split_value])

        node = np.array([feature, split_value, 1, left_tree.shape[0] + 1])

        return np.row_stack((node, left_tree, right_tree))

    def _get_best_feature(self, data_x, data_y):
        best_feature = 0
        best_correlation = -1
        for feature in range(data_x.shape[1]):
            correlation = np.corrcoef(data_x[:, feature], data_y)[0, 1] if np.std(data_x[:, feature]) > 0 else 0
            if correlation > best_correlation:
                best_correlation = correlation
                best_feature = feature
        return best_feature


if __name__ == "__main__":
    print("the secret clue is 'adwivedi62'")