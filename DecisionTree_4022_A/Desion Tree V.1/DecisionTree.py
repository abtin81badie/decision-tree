
import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm
import os 
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.left is None and self.right is None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, mode="entropy", random_choice=True):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.mode = mode
        self.random_choice = random_choice

    def fit(self, X, Y):
        if self.n_features is None or X.shape[1] > self.n_features:
            self.n_features = X.shape[1]
        self.root = self._build_tree(X, Y, [], 0)

    def _build_tree(self, X, Y, used_features, depth):
        num_samples, n_features = X.shape
        num_unique_labels = np.unique(Y)

        if (depth >= self.max_depth or len(num_unique_labels) == 1 or num_samples < self.min_samples_split or len(used_features) == n_features):
            leaf_value = self._most_common_value(Y)
            return Node(value=leaf_value)

        if self.random_choice:
            available_features = list(set(range(n_features)) - set(used_features))
            if len(available_features) == 0:
                leaf_value = self._most_common_value(Y)
                return Node(value=leaf_value)
            feature_indexs = np.random.choice(available_features, self.n_features, replace=True)
        else:
            feature_indexs = list(set(range(n_features)) - set(used_features))

        best_feature, best_thresh_hold = self._best_split(X, Y, feature_indexs)
        used_features.append(best_feature)

        left_index, right_index = self._split(X[:, best_feature], best_thresh_hold)

        if len(left_index) == 0:
            left_index = right_index

        if len(right_index) == 0:
            right_index = left_index

        left = self._build_tree(X[left_index, :], Y[left_index], used_features, depth+1)
        right = self._build_tree(X[right_index, :], Y[right_index], used_features, depth+1)

        return Node(best_feature, best_thresh_hold, left, right)

    def _most_common_value(self, Y):
        counter = Counter(Y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, Y, feature_indexs):
        best_gain = -100
        split_index, split_value = None, None

        for inx in feature_indexs:
            x_col = X[:, inx]

            if isinstance(x_col[0], (str, np.string_)):
                unique_values = np.unique(x_col)
                for value in unique_values:
                    gain = self._information_gain(Y, x_col, value)

                    if gain > best_gain:
                        best_gain, split_index, split_value = gain, inx, value
            else:
                thresholds = np.unique(x_col)
                for threshold in thresholds:
                    gain = self._information_gain(Y, x_col, threshold)

                    if gain > best_gain:
                        best_gain, split_index, split_value = gain, inx, threshold

        return split_index, split_value

    def _entropy(self, Y):
        counts = np.bincount(Y)
        probabilities = counts / len(Y)
        entropy = 0

        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _gini_index(self, Y):
        counts = np.bincount(Y)
        probabilities = counts / len(Y)
        gini = 1.00
        for p in probabilities:
            if p > 0:
                gini -= p**2
        return gini

    def _split(self, x_col, threshold):
        if isinstance(x_col[0], (str, np.string_)):
            left_index = np.argwhere(x_col == threshold).flatten()
            right_index = np.argwhere(x_col != threshold).flatten()
        else:
            left_index = np.argwhere(x_col <= threshold).flatten()
            right_index = np.argwhere(x_col > threshold).flatten()

        return left_index, right_index

    def _information_gain(self, Y, x_col, threshold):
        left_index, right_index = self._split(x_col, threshold)
        if len(left_index) == 0 or len(right_index) == 0: return 0

        n_left = len(left_index)
        n_right = len(right_index)
        n_total = n_left + n_right

        if self.mode == 'entropy':
            parent_entropy = self._entropy(Y)
            child_entropy = (n_left / n_total) * self._entropy(Y[left_index]) + (n_right / n_total) * self._entropy(Y[right_index])
            return parent_entropy - child_entropy
        else:
            gini = self._gini_index(Y) - ((n_left / n_total)*self._gini_index(Y[left_index]) + (n_right / n_total)*self._gini_index(Y[right_index]))
            return gini

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{indent}Leaf node with value: {node.value}")
        else:
            print(f"{indent}Split on feature {node.feature} <= {node.threshold}:")
            self.print_tree(node.left, indent + "-")
            self.print_tree(node.right, indent + "*")
if __name__== "__main__":
    df = pd.read_csv("onlinefraud.csv").dropna().drop("nameOrig", axis=1)
    x_t=df[{'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest','oldbalanceDest', 'newbalanceDest'}].to_numpy()
    y_t=df['isFraud'].to_numpy()
    X_train=x_t[:2000,:]
    y_train=y_t[:2000]
    X_test = x_t[2000:, :]
    y_test = y_t[2000:]
    print(x_t.shape,y_t.shape)
    print(X_train.shape,y_train.shape)
    
    clf = DecisionTree(max_depth=10,n_features=8)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
    
    acc = accuracy(y_test, predictions)
    print(acc)
# clf.print_tree()


# # def save_tree_to_dot(tree, feature_names, file_path='decision_tree.dot'):
# #     with open(file_path, 'w') as f:
# #         f.write('digraph Tree {\n')
# #         _tree_to_dot(tree.root, feature_names, f)
# #         f.write('}')

# # def _tree_to_dot(node, feature_names, f, indent=''):
# #     if node.is_leaf_node():
# #         f.write(f'{indent}{node.value}\n')
# #     else:
# #         if isinstance(node.threshold, str):
# #             condition = f'{feature_names[node.feature]} == "{node.threshold}"'
# #         else:
# #             condition = f'{feature_names[node.feature]} <= {node.threshold}'
# #         f.write(f'{indent}{feature_names[node.feature]} <= {node.threshold};\n')
# #         f.write(f'{indent}{condition} -> ')
# #         _tree_to_dot(node.left, feature_names, f, indent + '  ')
# #         f.write(f'{indent}{condition} -> ')
# #         _tree_to_dot(node.right, feature_names, f, indent + '  ')
# # def visualize_tree(tree, feature_names):
# #     dot_data = tree_to_dot(tree.root, feature_names)
# #     graph = graphviz.Source(dot_data)
# #     graph.render('decision_tree', format='png', cleanup=True)
# #     return graph

# # def tree_to_dot(node, feature_names):
# #     dot = []
# #     if node.is_leaf_node():
# #         dot.append(f"class={node.value}")
# #     else:
# #         if isinstance(node.threshold, str):
# #             condition = f'{feature_names[node.feature]} = "{node.threshold}"'
# #         else:
# #             condition = f'{feature_names[node.feature]} <= {node.threshold}'
# #         dot.append(f'"{condition}" [shape=box]')
# #         dot.append(f'{feature_names[node.feature]} -> "{condition}"')
# #         left = tree_to_dot(node.left, feature_names)
# #         right = tree_to_dot(node.right, feature_names)
# #         dot += left + right
# #     return '\n'.join(dot)

# df = pd.read_csv("onlinefraud.csv").dropna().drop("nameOrig", axis=1)
# x_t = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']].to_numpy()
# y_t = df['isFraud'].to_numpy()
# X_train = x_t[:2000, :]
# y_train = y_t[:2000]

# clf = DecisionTree(max_depth=10, n_features=8)
# clf.fit(X_train, y_train)

# # feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
# # # tree=tree_to_dot(clf, feature_names)
# # visualize_tree(clf,feature_names)