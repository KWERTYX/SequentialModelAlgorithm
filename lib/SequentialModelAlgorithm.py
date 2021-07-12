from sklearn import tree, preprocessing, model_selection
from sklearn.metrics import balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor

# Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier


import sklearn.utils as skl
import pandas as pd
import numpy as np
import random

class SequentialModelAlgorithm:
    # Se inicializan los hiper-parámetros como atributos dentro de la clase del problema
    # @ filename Nombre del fichero a elegir como dataset. Se carga cuando se inicia el programa con el método self.start()
    # @ nmodels cantidad de árboles a secuencializar
    # @ sample_size como proporción de ejemplos del conjunto de datos para entrenar cada árbol secuencial
    # @ max_depth como profundidad máxima para el entrenamiento de los árboles de decisión
    # @ lr como factor de aprendizaje
    def __init__(self, nmodels=300, sample_size=0.65, max_depth=10, lr=0.1, min_samples_leaf=1, max_features=None, min_weight_fraction_leaf=0.0, method="tree", neighbors=2, random=False):
        self.nmodels = nmodels
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.lr = lr
        self.pred = [0]
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.method = method
        self.neighbors = neighbors
        self.random = random
    
    # Comienza el proceso de creación de árboles 
    # @ attributes_cols columnas de atributos
    # @ objetive_col columna objetivo
    # @ random_state variable opcional que sirve de semilla para el primer split de datos
    # @ ftest_size variable opcional que sirve para elegir la proporción del primer split de datos
    def start(self, attributes_cols, objetive_col, random_state=12345, ftest_size=0.33):
        # Selección de columnas para atributos y objetivo
        attributes = self.attributes_preprocess(attributes_cols)
        objetive = self.objetive_preproccess(objetive_col)
        
        # Hacemos nuestra primera predicción sobre los resultados medios de cada columna
        cols_mean = attributes.mean(axis=0)
        
        # Realizamos la primera muestra
        (X_train, X_test, y_train, y_test) = model_selection.train_test_split(
                attributes, objetive,
                random_state=12345,
                test_size=ftest_size,
                stratify=objetive)

        if self.method == 'tree':
            model = tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, min_weight_fraction_leaf=self.min_weight_fraction_leaf)
        elif self.method == 'knn':
            model = KNeighborsRegressor(n_neighbors=self.neighbors)
        elif self.method == 'gradient':
            model = make_pipeline(StandardScaler(), SGDClassifier())
            

        #Entrenamos nuestra primera iteración
        model = model.fit(X_train, y_train)
        self.pred = model.predict([cols_mean])

        #print(Tree.score(X_test, y_test))
        
        models = []
        for i in range(self.nmodels):
            submodel, self.pred = self.meta_algorithm(X_test, y_test, self.pred)
            models.append(submodel)
            
        return models, balanced_accuracy_score(y_test, self.classify_prediction(self.pred)) # realmente debería ser self.classify_prediction(self.pred)
          
    # Preprocesado de atributos. Codifica datos en formato de texto en numéricos  
    # @ attributes_cols columnas de atributos
    def attributes_preprocess(self, attributes_cols):
        # Acumulamos las transformaciones necesarias en una matriz para realizarlas en la función ColumnTransformer
        transforms = []
        for i in attributes_cols: # Recorremos cada columna
            # Primer valor de cada columna. Podríamos recorrer cada valor de la columna para asegurarnos de que sean
            # numéricos o cadena de texto, pero en los Dataset de ejemplo nos vale con comprobar el primer valor.
            fval = attributes_cols[i][0]

            # Comprobamos que sea numérico, sino acumulamos una transformación sobre la columna
            if isinstance(fval, (int, float, np.int64))==False:
                transforms.append(("encode-"+i, preprocessing.OrdinalEncoder(), [i]))

        # Devolvemos como respuesta un array de enteros aplicando las transformaciones acumuladas en transforms
        # # remainder = 'passthrough' nos devolverá en la transformación también las columnas no afectadas
        res = np.array(ColumnTransformer(transforms, remainder='passthrough').fit_transform(attributes_cols), dtype=float)

        return res
    
    # Preprocesado de la columna objetivo. Codifica datos en formato de texto en numéricos  
    # @ objetive_col columnas objetivo
    def objetive_preproccess(self, objetive_col):
        # Primer valor de cada columna. Podríamos recorrer cada valor de la columna para asegurarnos de que sean
        # numéricos o cadena de texto, pero en los Dataset de ejemplo nos vale con comprobar el primer valor.
        fval = objetive_col[0]

        if isinstance(fval, (int, float, np.int64))==False:
            objetive_col = preprocessing.LabelEncoder().fit_transform(objetive_col)

        return objetive_col
    
    # Meta-algoritmo
    def meta_algorithm(self, X, y, prediction):        
        # residuoi
        i_res = y - prediction;

        Xm, i_resm = self.sample_without_replacement(X, i_res, self.sample_size)

        if self.random == False:
            # Creamos un nuevo modelo que entrenamos con la muestra y su residuo
            if self.method == 'tree':
                submodel = tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, min_weight_fraction_leaf=self.min_weight_fraction_leaf)
            elif self.method == 'knn':
                submodel = KNeighborsRegressor(n_neighbors=self.neighbors)
            elif self.method == 'gradient':
                submodel = make_pipeline(StandardScaler(), SGDClassifier())
        else:
            submodel = random.choice([tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, min_weight_fraction_leaf=self.min_weight_fraction_leaf), KNeighborsRegressor(n_neighbors=self.neighbors)])

        submodel = submodel.fit(Xm, i_resm)
        i_prediction = prediction + submodel.predict(X)*self.lr

        # Añadimos el nuevo modelo a la variable respuesta
        return submodel, i_prediction
        
    # Realiza un muestreo aleatorio de proporción sample_size del conjunto de datos y su residuo
    def sample_without_replacement(self, test_set, res_set, sample_size):
        # ordena aleatoriamente las dos matrices respetando que coincidan los índices
        test_set, res_set = skl.shuffle(test_set, res_set)

        # Limitamos la proporción de ejemplos según sample_size
        limit = int(test_set.shape[0]*sample_size)

        # Limitamos las matrices
        sample = test_set[0:limit]
        res = res_set[0:limit]

        return sample, res
    
    def classify_prediction(self, pred):
        Xmax = np.max(pred)
        Xmin = np.min(pred)
        for i in range(len(pred)):
            pred[i] = np.around((pred[i] - Xmin)/(Xmax-Xmin))
        return pred