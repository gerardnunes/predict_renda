import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import patsy
import statsmodels.api as sm

# Título
st.title('Análise de Renda')

renda = pd.read_csv('pv_de_renda.csv')
rendafil = renda.dropna().drop(columns=['id_cliente', 'Unnamed: 0'], errors='ignore')

st.sidebar.header("Seleção de Página")
pagina = st.sidebar.selectbox("Escolha uma página:", ["Análises", "Modelos"])

# Página de Análises
if pagina == "Análises":
    st.header("Análises Univariadas e Multivariadas")

    st.subheader("1. Análise Univariada")
    opc = ['id_cliente', 'qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
    coluna = st.selectbox("Escolha uma coluna para o gráfico:", opc)
    grafico = st.selectbox("Escolha o tipo de gráfico:", ['histograma', 'boxplot'])

    var = renda[coluna]

   
    plt.figure(figsize=(8, 5))
    if grafico == 'histograma':
        plt.hist(var, bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {coluna.capitalize()}')
        plt.xlabel(coluna.capitalize())
        plt.ylabel('Frequência')
    else:
        sns.boxplot(x=var, color='lightgreen')
        plt.title(f'Boxplot de {coluna.capitalize()}')
        plt.xlabel(coluna.capitalize())

    st.pyplot(plt)

    st.subheader('2. Visualização dos Dados')
    st.write(renda.head())

  
    st.subheader('3. Matriz de Correlação')
    numericas = renda.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = renda[numericas].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação')
    st.pyplot(plt)
    
#pagina modelos

elif pagina == "Modelos":
    st.header("Modelos Preditivos para Análise de Renda")
    
    modelo_selecionado = st.selectbox("Escolha um modelo:", ["Lasso", "OLS", "Random Forest"])
    grafico_selecionado = st.selectbox("Escolha um tipo de gráfico:", ["Gráfico de Resíduos", "Histograma de Predições"])

    y, X = patsy.dmatrices('np.log(renda) ~ sexo + posse_de_imovel + qtd_filhos + idade + tempo_emprego', rendafil)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if modelo_selecionado == "Lasso":
        st.subheader("Modelo Lasso")
        modelo_lasso = Lasso(alpha=0.1).fit(X_train, Y_train)
        predicoes = modelo_lasso.predict(X_test)
    elif modelo_selecionado == "OLS":
        st.subheader("Modelo OLS")
        modelo_ols = sm.OLS(Y_train, X_train).fit()
        predicoes = modelo_ols.predict(X_test)
    elif modelo_selecionado == "Random Forest":
        st.subheader("Modelo Random Forest")
        regressor = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, Y_train.ravel())
        predicoes = regressor.predict(X_test)

    r2 = r2_score(Y_test, predicoes)
    st.write(f"R² do modelo ({modelo_selecionado}): {r2:.4f}")
    
    Y_test_flat = Y_test.ravel()
    residuos = Y_test_flat - predicoes
    
    plt.figure(figsize=(10, 5))
    if grafico_selecionado == "Gráfico de Resíduos":
        sns.scatterplot(x=Y_test_flat, y=residuos, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Resíduos')
        plt.title(f'Gráfico de Resíduos - {modelo_selecionado}')
    else:
        plt.hist(predicoes, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Predições')
        plt.ylabel('Frequência')
        plt.title(f'Histograma de Predições - {modelo_selecionado}')

    st.pyplot(plt)
