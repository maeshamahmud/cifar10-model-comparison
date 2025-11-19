import numpy as np


class ManualDecisionTree:
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    # ---------- Public API ---------- #

    def fit(self, X, y):
        """
        X: (N, D) NumPy array
        y: (N,)   NumPy array of labels
        """
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        X: (N, D) NumPy array
        Returns: (N,) predicted labels
        """
        return np.array([self._predict_one(x, self.tree_) for x in X])

    # ---------- Tree building ---------- #

    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        num_labels = len(np.unique(y))

        # Stopping conditions: max depth, pure node, or too few samples
        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return {
                "leaf": True,
                "value": leaf_value,
            }

        # Find best split
        best_feat, best_thresh, best_gain, best_sets = self._best_split(X, y)

        # If no good split found, make leaf
        if best_feat is None or best_gain <= 0.0 or best_sets is None:
            leaf_value = self._most_common_label(y)
            return {
                "leaf": True,
                "value": leaf_value,
            }

        X_left, y_left, X_right, y_right = best_sets

        # Recursively build children
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": left_child,
            "right": right_child,
        }

    def _best_split(self, X, y):
        """
        Try all features and candidate thresholds, pick the one
        with maximum Gini information gain.
        """
        m, n_features = X.shape
        if m <= 1:
            return None, None, 0.0, None

        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        best_sets = None

        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]

            # Unique sorted values for this feature
            values = np.unique(X_col)
            if values.shape[0] == 1:
                continue  # no split possible

            # Candidate thresholds: midpoints between consecutive unique values
            thresholds = (values[:-1] + values[1:]) / 2.0

            for t in thresholds:
                left_mask = X_col <= t
                right_mask = X_col > t

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                gain = self._information_gain(parent_gini, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = t
                    best_sets = (
                        X[left_mask],
                        y_left,
                        X[right_mask],
                        y_right,
                    )

        return best_feat, best_thresh, best_gain, best_sets

    # ---------- Impurity & gain ---------- #

    def _gini(self, y):
        """
        Gini impurity of a label array y.
        Gini = 1 - sum_k (p_k)^2
        """
        m = len(y)
        if m == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / m
        return 1.0 - np.sum(probs ** 2)

    def _information_gain(self, parent_gini, y_left, y_right):
        """
        Information gain from splitting parent set into y_left and y_right.
        """
        m = len(y_left) + len(y_right)
        if m == 0:
            return 0.0
        g_left = self._gini(y_left)
        g_right = self._gini(y_right)
        weighted_gini = (len(y_left) / m) * g_left + (len(y_right) / m) * g_right
        return parent_gini - weighted_gini

    # ---------- Utils ---------- #

    def _most_common_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
