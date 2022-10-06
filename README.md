# CS-422-Decision-Trees-and-Random-Forests
Implementation of a ML Decision Tree model using an information gain based algorithm and
then builds upon this model to implement a Random Forest algorithm

The data_storage.py file defines two classes, Binary Tree and Node. Both
serve as the decision trees data storage model.

The decision_trees.py file defines various functions however the four most important
are: DT_train_binary(), DT_make_prediction(), RF_build_random_forest(), and RF_test_random_forest().

Both DT_train_binary and RF_build_random_forest build and train a decision tree based on information
gain and on the random set of data, in the case of the random forest.

DT_make_prediction and RF_build_random_forest both take a completed decision, or set of decisions trees,
and produces a prediction based off some data input.
