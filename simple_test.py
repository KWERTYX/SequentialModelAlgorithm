# SequentialModelAlgorith Class
from lib.SequentialModelAlgorithm import SequentialModelAlgorithm

import pandas as pd
import numpy as np

# We fisrt select a dataset
dataset = pd.read_csv('./datasets/adultDataset.csv', header = 0)
# Then we chose its attribute columns and its objetive column
attr_cols = dataset.loc[:, 'capital-gain':'native-country']
obj_col = dataset['income']

# The default values for the main hyperarguments. There are arguments to change, and this can optimize the problem scenario.
model = SequentialModelAlgorithm(nmodels=300, sample_size=0.65, max_depth=10, lr=0.1)
submodels, score = model.start(attributes_cols = attr_cols, objetive_col = obj_col)

# The score evaluated with BalancedAccuracyScore from sklearn from the last submodel
print('The BalancedAccuracyScore is: '+str(score))