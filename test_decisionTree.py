import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lib.SequentialModelAlgorithm import SequentialModelAlgorithm

dataset = pd.read_csv('./datasets/titanic.csv', header = 0)
attr_cols = dataset.loc[:, 'Pclass':'Is_Married']
obj_col = dataset['Survived']

values = [1, 25, 50, 100, 150, 250, 350, 400]
print('\n--- Increasing nmodels ['+str(np.min(values))+' - '+str(np.max(values))+'] ---')
scores = []

for v in range(len(values)):
    submodel = SequentialModelAlgorithm(nmodels = values[v], method = "tree")
    submodels, score = submodel.start(attributes_cols = attr_cols, objetive_col = obj_col)
    scores.append(score)
    
# Realizamos una gráfica para determinar cómo afecta ntree a la puntuación
plt.plot(values, scores)
plt.xlabel('- nmodels -')
plt.ylabel('- Balanced Accuracy Score -')
plt.title('increaing nmodel hyperargument')
plt.show()
m = np.argmax(scores)
print('Mejor puntuación: '+str(scores[m])+' con valor = '+str(values[m]))