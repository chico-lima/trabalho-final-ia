import streamlit as st
import modelos.modelo as modelo1
import modelos.modelo2 as modelo2
import modelos.modelo3 as modelo3
import modelos.modelo4 as modelo4
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão FIPE", layout="wide")

st.title("Previsão de Preços FIPE - Modelos de Regressão Linear")

# =========================
# 1. Menu lateral
# =========================
menu = ["Modelo 1", "Modelo 2", "Modelo 3", "Modelo 4"]
escolha = st.sidebar.selectbox("Escolha o modelo:", menu)

# =========================
# 2. Inputs do usuário
# =========================
st.sidebar.header("Parâmetros do carro")
marca = st.sidebar.text_input("Marca", "Acura")
modelo_carro = st.sidebar.text_input("Modelo", "Legend 3.2/3.5")
ano_modelo = st.sidebar.number_input("Ano do modelo", min_value=1920, max_value=2030, value=1998)
preco_real_input = st.sidebar.number_input("Preço real (opcional)", value=27666.0, step=100.0)

# =========================
# 3. Carregar dados e rodar modelo
# =========================
if escolha == "Modelo 1":
    df = modelo1.carregar_dados()
    res, erro = modelo1.executar_modelo(df, marca, modelo_carro, ano_modelo)
    plot_func = modelo1.plot_regressao

elif escolha == "Modelo 2":
    df = modelo2.carregar_dados()
    res, erro = modelo2.executar_modelo(df, marca, modelo_carro, ano_modelo)
    plot_func = modelo2.plot_regressao

elif escolha == "Modelo 3":
    df = modelo3.carregar_dados()
    res, erro = modelo3.executar_modelo(df, marca, modelo_carro, ano_modelo, preco_real_input)
    plot_func = modelo3.plot_regressao

elif escolha == "Modelo 4":
    df = modelo4.carregar_dados()
    res, erro = modelo4.executar_modelo(df, marca, modelo_carro, ano_modelo, preco_real_input)
    plot_func = modelo4.plot_regressao


# =========================
# 4. Mostrar resultados
# =========================
if erro:
    st.error(erro)

else:
    st.subheader(f"Resultado - {escolha}")

    # Campos básicos
    st.markdown(f"<h3><b>Marca:</b> {marca}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3><b>Modelo:</b> {modelo_carro}</h3>", unsafe_allow_html=True) 
    st.markdown(f"<h3><b>Ano do modelo:</b> {ano_modelo}</h3>", unsafe_allow_html=True)


    # Preço previsto (se existir)
    if "preco_previsto" in res:
        st.markdown(
            f"<h3><b>Preço previsto:</b> R$ {res['preco_previsto']:.2f}</h3>",
            unsafe_allow_html=True
        )

    # Preço real (se existir)
    if "preco_real" in res:
        st.markdown(
            f"<h3><b>Preço real:</b> R$ {res['preco_real']:.2f}</h3>",
            unsafe_allow_html=True
        )

    # Métricas — somente se existirem
    if "mae" in res:
        st.markdown(f"<h3><b>MAE:</b> {res['mae']:.2f}</h3>", unsafe_allow_html=True)
    if "mse" in res:
        st.markdown(f"<h3><b>MSE:</b> {res['mse']:.2f}</h3>", unsafe_allow_html=True)
    if "rmse" in res:
        st.markdown(f"<h3><b>RMSE:</b> {res['rmse']:.2f}</h3>", unsafe_allow_html=True)
    if "erro_percentual" in res:
        st.markdown(
            f"<h3><b>Erro percentual:</b> {res['erro_percentual']:.2f}%</h3>",
            unsafe_allow_html=True
        )

    # Gráfico
    fig = plot_func(res)
    st.pyplot(fig)



