# Projeto-Indicium para ciência de dados

Este projeto utiliza um modelo de árvore de decisão (regressão) para prever a nota IMDB de filmes. Inclui um Jupyter Notebook para treinamento e teste dos dados e um arquivo `.pkl` com o modelo treinado.

## Requisitos

- Python 3.7 ou superior
- Jupyter Notebook

## Requisitos opcionais

- Bibliotecas Python listadas em `requirements.txt`

## Instalação (opcional)

1. **Clone o repositório**:
    ```bash
    git clone https://github.com/GCurvo/projeto-indicium.git
    cd projeto-indicium
    ```

2. **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

## Executando o Jupyter Notebook

1. **Inicie o Jupyter Notebook**:

2. **Abra o arquivo `indicium_arvore_de_decisao.ipynb`**:
    - No navegador, navegue até o diretório onde você clonou o repositório.
    - Clique no arquivo `indicium_arvore_de_decisao.ipynb` para abri-lo.

3. **Carregue a base de dados e a previsão**:
    - Pegue os arquivos `desafio_indicium_imdb.csv` e o `previsao.csv` e deixe eles na área do arquivo para ser carregado pelo código.
    - Verifique se o caminho para o arquivo esteja correto.

4. **Execute as células do notebook**:
    - Siga as instruções no notebook para carregar os dados, treinar o modelo e realizar análises.

## Carregando e Utilizando o Modelo Treinado (`modelo_arvore.pkl`)

O arquivo `modelo_arvore.pkl` contém o modelo treinado. Aqui está um exemplo de como carregar e utilizar este modelo em um script Python.

### Exemplo de Uso

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


file_path=('/content/previsao.csv')

df = pd.read_csv(file_path)

prediction_data = df

# Carregando o modelo treinado
with open("modelo_arvore.pkl", "rb")as arquivo:
  model = pickle.load(arquivo)

# Aplicar as mesmas transformações aos dados de previsão
prediction_data.drop(["Director", "Star1", "Star2", "Star3", "Star4", "Overview", "Series_Title"], axis=1, inplace=True)

label_encoder = LabelEncoder()

# Transformar colunas categóricas usando LabelEncoder
categorical_columns = ['Genre', 'Certificate', 'Released_Year', 'Gross', 'Runtime']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Fazer previsões
predictions = model.predict(prediction_data)

# Arredondar previsões para duas casas decimais
predictions = np.round(predictions, 2)

print(predictions)
