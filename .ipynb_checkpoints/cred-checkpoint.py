import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from ydata_profiling import ProfileReport
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression


st.title('Análise de Renda')

renda = pd.read_csv('pv_de_renda.csv')

st.write("analise univariada.")
st.sidebar.header("Seleção de Período e Operação")
opg= [
    'histograma',
    'boxplot'
]
opc=
grafico = selecao = st.selectbox("coluna Y")
coluna = selecao = st.selectbox("coluna Y")

idade = df['idade']

# Histograma
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # Área para o histograma
plt.hist(idade, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')

# Boxplot
plt.subplot(1, 2, 2)  # Área para o boxplot
sns.boxplot(data=idade, orient='h', color='lightgreen')
plt.title('Boxplot de Idade')
plt.xlabel('Idade')

plt.tight_layout()  # Ajusta o layout
plt.show()








st.subheader('Visualização dos Dados')
st.write(renda.head())


numericas = renda.select_dtypes(include=['int64', 'float64']).columns


correlation_matrix = renda[numericas].corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')

# Mostrar o gráfico no Streamlit
st.pyplot(plt)

# Opcional: adicionar uma descrição
st.write("Essa matriz de correlação mostra a relação entre as variáveis numéricas no conjunto de dados.")





rendafil = renda.dropna()

naco = rendafil.isna().sum()

rendafil = rendafil.drop(columns=['id_cliente','Unnamed: 0'])
rendafil.dtypes



























y, X = patsy.dmatrices('np.log(renda) ~ sexo  + posse_de_imovel + qtd_filhos  + idade + tempo_emprego', rendafil)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelos 


st.write('modelo ridge')
modelr = Ridge(alpha=1.0).fit(X_train, Y_train)


rid = modelr.predict(X_test)

ridge_r2 = r2_score(Y_test, rid)

print(f'Ridge Model R²: {ridge_r2:.4f}')

Y_test_flat = Y_test.ravel()  
rdp_flat = rid.flatten()     



ress = Y_test_flat - rdp_flat
plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_test_flat, y=ress, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
plt.show()
st.pyplot(plt)




# modelo 2
y, X = patsy.dmatrices('np.log(renda) ~ sexo  + posse_de_imovel + qtd_filhos  + idade + tempo_emprego', rendafil)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

p1model = sm.OLS(Y_train, X_train).fit()
rdp = p1model.predict(X_test)
r22 = r2_score(Y_test, rdp)
print(f'r2 {r22:.4f}')




Y_test_flat = Y_test.ravel()  
rdp_flat = rdp.flatten()     


residuos = Y_test_flat - rdp_flat


plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_test_flat, y=residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
st.pyplot(plt)


#modelo3
y, X = patsy.dmatrices('np.log(renda) ~ sexo  + posse_de_imovel + qtd_filhos  + idade + tempo_emprego', rendafil)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("arvore")
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # Usando mais árvores pode melhorar o desempenho
regressor.fit(X_train, Y_train.ravel())  # Use ravel() para converter Y_train para uma dimensão correta
predicoes = regressor.predict(X_test)

r2 = metrics.r2_score(Y_test, predicoes)
print(f'R²: {r2:.4f}')

residuos = Y_test.ravel() - predicoes

r2 = metrics.r2_score(Y_test, predicoes)
mae = metrics.mean_absolute_error(Y_test, predicoes)
mse = metrics.mean_squared_error(Y_test, predicoes)
rmse = mse ** 0.5  # Raiz do erro quadrático médio


plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_test.ravel(), y=residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
plt.show()
st.pyplot(plt)