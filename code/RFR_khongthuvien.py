import numpy as np


# Một nút trên cây: hoặc là nút lá (giá trị dự đoán), hoặc nút chia (feature + ngưỡng + 2 con).
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Cây hồi quy: tại mỗi bước chọn feature + ngưỡng để giảm phương sai; lá = trung bình y.
class DecisionTreeRegressorCustom:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape

        # Dừng khi đủ sâu / ít mẫu → lá trả về mean(y).
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Mỗi lần chia chỉ xét một tập con feature ngẫu nhiên (giống Random Forest).
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Không còn split hữu ích → lá.
        if best_feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_var_red = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                var_red = self._variance_reduction(y, X_column, thr)

                if var_red > best_var_red:
                    best_var_red = var_red
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _variance_reduction(self, y, X_column, threshold):
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Giảm phương sai = var(tổng) - trọng số hóa var(trái/phải).
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        var_y = np.var(y)
        var_l, var_r = np.var(y[left_idxs]), np.var(y[right_idxs])

        var_reduction = var_y - (n_l / n * var_l + n_r / n * var_r)
        return var_reduction

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# Rừng: nhiều cây, mỗi cây học trên bootstrap + subset feature; dự đoán = trung bình các cây.
class RandomForestRegressorCustom:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )

            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
