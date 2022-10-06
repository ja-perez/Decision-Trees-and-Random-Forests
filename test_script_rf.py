import numpy as np
import data_storage as ds
import decision_trees as dt

file_name = "haberman.csv"
data = np.genfromtxt(file_name, dtype=str, delimiter=',')
samples,labels = ds.build_nparray(data)

max_depth = 3
tree_count = 11
forest = dt.RF_build_random_forest(samples,labels,max_depth,tree_count)
test_accuracy = dt.RF_test_random_forest(samples,labels,forest)
print("RF: ",test_accuracy)