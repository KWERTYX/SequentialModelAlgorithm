# Python IA ensembling prototype to get more accuracy in predictions | SequentialModelAlgorithm
## Summary
This class works with different Learnings Model of type Regression in sklearn. The aim of the project is to make a harder learner model with the ensembling of multiple simpler learning submodels that will sequentially learn from the fail of each previous model of the process. The last submodel will give better and more accuracy predictions to evaluations.

Download the project folder and unzip. The class are located in the lib folder and are called from the Jupyter Notebooks and the Python tests in the root folder. There are 3 public datasets insite the datasets folder.

## SequentialModelAlgorithm
The class is going to recreate a first prototype problem of learning model. The hyperarguments for this supervised problem scene will be the constructor arguments of the class:
* nmodels: number of sequential models in the algorithm
* sample_size: proportion of samples to take from a random sample taken from the evaluation data set to train subsequential learning models
* max_depth [Decision Trees Model]: maximum distance between the root and any leaf in the decision tree
* lr: learning factor of each subsequential learning model of the previous ones
* max_features
* min_weight_fraction_leaf
* method: "tree" to use Regression Decision Trees or "knn" to use K-Nearest Neighbors

### Use example
Importing the classes and defining the input data:

```Python
# We will need this for sure ;)
import pandas as pd
import numpy as np

# SequentialModelAlgorith Class
from lib.SequentialModelAlgorithm import SequentialModelAlgorithm

# We fisrt select a dataset
dataset = pd.read_csv('./datasets/adultDataset.csv', header = 0)
# Then we chose its attribute columns and its objetive column
attr_cols = dataset.loc[:, 'capital-gain':'native-country']
obj_col = dataset['income']
```

Constructing the class and starting the subsequential learning process:
```Python
# The default values for the main hyperarguments. There are arguments to change, and this can optimize the problem scenario.
model = SequentialModelAlgorithm(nmodels=300, sample_size=0.65, max_depth=10, lr=0.1)
submodels, score = model.start(attributes_cols = attr_cols, objetive_col = obj_col)

# The score evaluated with BalancedAccuracyScore from sklearn from the last submodel
print('The BalancedAccuracyScore is: '+str(score))
```
Output:
> The BalancedAccuracyScore is: 0.9012394505275068
