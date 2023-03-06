import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class ClientBank:
    def __init__(self):
        # Lendo a base de dados
        self.df = pd.read_csv("exemplo2.csv")
    
        # Separando o input do output
        X = self.df.drop("risco", axis= 1)
        y = self.df.risco

        # Normalizando os dados
        self.minmax = MinMaxScaler()
        X_norm = self.minmax.fit_transform(X)

        # Gerando as variáveis de treino e teste
        X_train, self.X_test, y_train, self.y_test = train_test_split(X_norm, y, test_size= 1/3, random_state= 46)

        # Instanciando o algoritmo knn para 3 vizinhos
        self.knn = KNeighborsClassifier(n_neighbors= 3)

        # Treinando o modelo
        self.knn.fit(X_train, y_train)
    
    def predictClient(self, age: int, balance: float):
        # Normalizando os dados de entrada
        new_client_norm = self.minmax.transform([[age, balance]])

        # Gerando e retornando a variável do resultado
        result = self.knn.predict(new_client_norm)
        return result[0]
    
    def getMlScore(self):
        # Gerando e retornando a variável do score do modelo
        score = accuracy_score(self.y_test, self.knn.predict(self.X_test))
        return score