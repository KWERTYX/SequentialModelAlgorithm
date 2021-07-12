import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lib.SequentialDecisionTreeAlgorithm import SequentialDecisionTreeAlgorithm
from lib.SequentialModelAlgorithm import SequentialModelAlgorithm

dataset = pd.read_csv('./datasets/adultDataset.csv', header = 0)
attr_cols = dataset.loc[:, 'capital-gain':'native-country']
obj_col = dataset['income']

values = [5, 10, 15, 20]
print('\n--- Increasing nmodels ['+str(np.min(values))+' - '+str(np.max(values))+'] ---')

scores = []
for v in range(len(values)):
    model = SequentialModelAlgorithm(nmodels = values[v], method = "knn")
    submodels, score = model.start(attributes_cols = attr_cols, objetive_col = obj_col)
    scores.append(score)
    
# We plot a tree to deterinate how ntree affects to the final score
plt.plot(values, scores)
plt.xlabel('- nmodels -')
plt.ylabel('- Balanced Accuracy Score -')
plt.title('score depending on nmodels\' values')
plt.show()
m = np.argmax(scores)
print('Best score: '+str(scores[m])+' with value = '+str(values[m]))