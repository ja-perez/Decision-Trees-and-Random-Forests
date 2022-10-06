import numpy as np
import math


def calc_entropy(values):
    if values.size:
        h_one_count, h_zero_count = np.count_nonzero(values == 1), np.count_nonzero(values == 0)
        h_one_prob, h_zero_prob = h_one_count / np.size(values), h_zero_count / np.size(values)
        try:
            h_one = -1 * h_one_prob * math.log2(h_one_prob)
        except ValueError as _:
            h_one = 0
        try:
            h_zero = -1 * h_zero_prob * math.log2(h_zero_prob)
        except ValueError as _:
            h_zero = 0
        return h_one + h_zero
    else:
        return 0


id_nums = 0


class Node:

    def __init__(self, data, labels, h_base=-1):
        global id_nums
        self.data = data
        self.labels = labels
        self.h_base = h_base
        self.left, self.right = None, None
        self.prediction = 2
        self.information_gain = 0
        self.node_id = id_nums
        self.node_feature_index = None
        id_nums += 1

    def __str__(self):
        meta_data = "Data:\n" + str(self.data) + '\n' + "Labels:\t" + str(self.labels) + '\n'
        calc_data = "Entropy:" + str(self.h_base) + '\n' + "Information Gain: " + \
                    str(self.information_gain) + '\n' + "Prediction:" + str(self.prediction)
        return meta_data + '\n' + "Feature Index:" + str(self.node_feature_index) + '\n'

    def calc_information_gain(self, feature_index):
        zero_rows = [i for i, x in enumerate(self.data[:, feature_index]) if x == 0]
        one_rows = [i for i, x in enumerate(self.data[:, feature_index]) if x == 1]
        left_data = np.array([self.data[coord] for coord in zero_rows])
        left_labels = np.array([self.labels[coord] for coord in zero_rows])
        right_data = np.array([self.data[coord] for coord in one_rows])
        right_labels = np.array([self.labels[coord] for coord in one_rows])
        left_h = calc_entropy(left_labels)
        right_h = calc_entropy(right_labels)
        zero_prob = len(zero_rows) / len(self.data)
        one_prob = len(one_rows) / len(self.data)
        self.information_gain = self.h_base - sum([zero_prob * left_h, one_prob * right_h])
        return (left_data, left_labels), (right_data, right_labels)

    def add_feature(self, feature_index):
        left_node, right_node = self.calc_information_gain(feature_index)
        self.left = Node(left_node[0], left_node[1])
        self.right = Node(right_node[0], right_node[1])
        self.left.calculate_entropy(left_node[1])
        self.right.calculate_entropy(right_node[1])

    def calculate_entropy(self, labels):
        curr_entropy = calc_entropy(labels)
        if self.h_base == -1:
            self.h_base = curr_entropy
        return curr_entropy

    def gen_prediction(self):
        self.prediction = 1 if np.count_nonzero(self.labels == 1) >= np.count_nonzero(self.labels == 0) else 0


class BinaryTree:

    def __init__(self, data, labels, h_base, max_depth):
        self.data = data
        self.labels = labels
        self.h_base = h_base
        self.max_depth = max_depth
        self.root = Node(data, labels, h_base)
        self.height = 0
        self.features = []
        self.last_feature_index = -1

    def generate_tree(self, curr_node=None):
        # Left - Right - Root
        if not curr_node:
            if self.root:
                curr_node = self.root
            else:
                return
        if not curr_node.left and not curr_node.right:
            self.last_feature_index, max_ig, feature_index = -1, -1, 0
            # Leaf node - Add features here
            node_level = self.get_level(self.root, curr_node.node_id)
            if node_level + 1 <= self.max_depth and self.features and curr_node.h_base:
                for feature in self.features:
                    curr_node.calc_information_gain(feature)
                    if curr_node.information_gain > max_ig:
                        feature_index = feature
                        max_ig = curr_node.information_gain
                curr_node.add_feature(feature_index)
                curr_node.node_feature_index = feature_index
                self.height = self.get_height()
                self.last_feature_index = feature_index
            else:
                # Max depth or out of features - Finalize node
                self.height += 1
                curr_node.gen_prediction()
            return

        elif not curr_node.left:
            pass
        else:
            self.generate_tree(curr_node.left)
            if not curr_node.right:
                return
            else:
                self.generate_tree(curr_node.right)

    def post_order_traversal(self, curr_node=None):
        # Left - Right - Root
        if not curr_node:
            if not self.root:
                return
            curr_node = self.root
        if not curr_node.left and not curr_node.right:
            # Leaf node
            print("Leaf - Level:", self.calc_height(self.root))
            print(curr_node)
            return
        elif not curr_node.left:
            # Unsure about what this does or its importance
            print(curr_node)
        else:
            print("left")
            self.post_order_traversal(curr_node.left)
            if not curr_node.right:
                # No right node, print root and return to previous call
                print(curr_node)
            else:
                print("right")
                self.post_order_traversal(curr_node.right)
                # Root after traversing all right nodes
                print("root")
                print(curr_node)

    def in_order_traversal(self, curr_node=None):
        if not curr_node:
            if not self.root:
                return
            curr_node = self.root
        if not curr_node.left and not curr_node.right:
            print("Leaf")
            print(curr_node)
            return
        elif not curr_node.left:
            print(curr_node)
            self.in_order_traversal(curr_node.right)
        else:
            print("Left")
            self.in_order_traversal(curr_node.left)
            print("Root")
            print(curr_node)
            if not curr_node.right:
                return
            else:
                print("Right")
                self.in_order_traversal(curr_node.right)

    def specific_level(self, curr_node, level):
        if not curr_node:
            return
        if level == 1:
            print(curr_node)
            print()
        elif level > 1:
            self.specific_level(curr_node.left, level - 1)
            self.specific_level(curr_node.right, level - 1)

    def level_order_traversal(self):
        for i in range(self.calc_height(self.root) + 1):
            self.specific_level(self.root, i)

    def calc_height(self, curr_node):
        if not curr_node:
            return 0
        else:
            l_height = self.calc_height(curr_node.left)
            r_height = self.calc_height(curr_node.right)
            if l_height > r_height:
                return l_height + 1
            else:
                return r_height + 1

    def get_height(self):
        return self.calc_height(self.root)

    def get_node_level(self, curr_node, node_id, level):
        if not curr_node:
            return 0
        if curr_node.node_id == node_id:
            return level
        next_level = self.get_node_level(curr_node.left, node_id, level + 1)
        if next_level:
            return next_level
        next_level = self.get_node_level(curr_node.right, node_id, level + 1)
        return next_level

    def get_level(self, curr_node, node_id):
        return self.get_node_level(curr_node, node_id, 1)

    def get_prediction(self, curr_node, test_data):
        if curr_node.prediction != 2:
            return curr_node.prediction
        if not curr_node:
            if not self.root:
                return
            curr_node = self.root
        feature_index = curr_node.node_feature_index
        if np.any(test_data[feature_index]):
            return self.get_prediction(curr_node.right, test_data)
        else:
            return self.get_prediction(curr_node.left, test_data)


def build_nparray(data):
    samples = np.array(data[1:, :-1], float)
    labels = np.array(data[1:, -1], int)
    return samples, labels


def build_list(data):
    features = []
    label_data = []
    for points in data[1:]:
        feature_data = []
        for point in points[:-1]:
            feature_data.append(float(point))
        label_data.append(int(points[-1]))
        features.append(feature_data)
    return features, label_data


def build_dict(data):
    features = []
    samples = {}
    labels = {}
    for feature in data[0]:
        features.append(feature)
    for i, points in enumerate(data[1:]):
        sample = {}
        for j, point in enumerate(points[:-1]):
            curr_feature = features[j]
            sample[curr_feature] = float(point)
        samples[i] = sample
        labels[i] = int(points[-1])
    return samples, labels
