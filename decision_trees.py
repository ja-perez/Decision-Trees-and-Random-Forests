import data_storage as ds
import numpy as np


def DT_train_binary(x, y, max_depth):
    # X (n.array) - Feature Training Data
    # Y (n.array) - Label Training Data
    # ig / IG - Information Gain

    features = [index for index in range(len(x[0]))]
    h_base = ds.calc_entropy(y)
    training_dt = ds.BinaryTree(x, y, h_base, max_depth)
    training_dt.features = features

    if max_depth != -1:
        tree_height = 0
        while tree_height <= max_depth + 1:
            training_dt.generate_tree()
            tree_height = training_dt.height
            if training_dt.last_feature_index != -1:
                training_dt.features.remove(training_dt.last_feature_index)
    else:
        training_dt.max_depth = len(features) + 1
        tree_height = 0
        while tree_height <= training_dt.max_depth and features:
            training_dt.generate_tree()
            tree_height = training_dt.height
            if training_dt.last_feature_index != -1:
                training_dt.features.remove(training_dt.last_feature_index)
    # training_dt.level_order_traversal()
    return training_dt


def DT_test_binary(x, y, dt):
    # Uses DT_train_binary prediction to measure accuracy
    # X - Feature Test Data
    # Y - Label Test Data
    # Calculate Label
    outcomes = []
    for i, row in enumerate(x):
        prediction = dt.get_prediction(dt.root, row)
        outcomes.append(prediction == y[i])
    return sum(outcomes) / len(outcomes)


def DT_make_prediction(x, dt):
    return dt.get_prediction(dt.root, x)


def RF_build_random_forest(x, y, max_depth, num_of_trees):
    forest = []
    combined_data = np.concatenate((x, np.array([y]).T), axis=1)
    sample = np.random.default_rng()
    for _ in range(num_of_trees):
        random_sample = sample.choice(combined_data, int(len(x) * .1), replace=False)
        random_x, random_y = random_sample[:, :-1], random_sample[:, -1]
        forest.append(DT_train_binary(random_x, random_y, max_depth))
    return forest


def RF_test_random_forest(x, y, rf):
    outcomes = []
    accuracies = []
    for i, tree in enumerate(rf):
        for j, row in enumerate(x):
            prediction = tree.get_prediction(tree.root, row)
            outcomes.append(prediction == y[j])
        accuracies.append(DT_test_binary(x, y, tree))
        print("DT", i, ":", round(accuracies[-1], 5))
    print(outcomes)
    return round(sum(outcomes) / len(outcomes), 5)
