import numpy as np
import random


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose
        random.seed(903862212)

    def author(self):
        return 'aishwaryD'

    def add_evidence(self, data_x, data_y):
        self.tree = self._build_tree(data_x, data_y)

        if self.verbose:
            print("RTLearner")
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

        feature = self._get_best_feature(data_x)
        random1, random2 = np.random.choice(data_x.shape[0], 2, replace=False)
        split_value = (data_x[random1, feature] + data_x[random2, feature]) / 2

        if np.allclose((data_x[:, feature] <= split_value), (data_x[:, feature] <= split_value)[0]):
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])

        left_tree = self._build_tree(data_x[data_x[:, feature] <= split_value], data_y[data_x[:, feature] <= split_value])
        right_tree = self._build_tree(data_x[data_x[:, feature] > split_value], data_y[data_x[:, feature] > split_value])

        node = np.array([feature, split_value, 1, left_tree.shape[0] + 1])

        return np.row_stack((node, left_tree, right_tree))

    def _get_best_feature(self, data_x):
        num_features = data_x.shape[1]
        feature = np.random.choice(num_features)
        return feature


if __name__ == "__main__":
    print("the secret clue is 'adwivedi62'")
