import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==============================
#  FUNÇÃO 1 — CARREGAR DATASET
# ==============================
def carregar_dados(caminho="/dataset/fipe_2022.csv"):
    df = pd.read_csv(caminho)

    df = df.drop(["fipe_code","authentication","gear"], axis=1)
    df = df.drop_duplicates()

    map_meses = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    df["month_of_reference"] = df["month_of_reference"].map(map_meses)
    df["year_of_reference"] = df["year_of_reference"].astype(int)

    df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

    df = df.sort_values(["brand", "model", "year_of_reference", "month_of_reference"])
    return df


# ======================================
#  FUNÇÃO 2 — EXECUTAR O MODELO 1
# ====================================== 
def executar_modelo(df, marca, modelo_carro, ano_modelo):
    # Filtrar apenas o modelo desejado
    df_modelo = df[(df["brand"] == marca) & (df["model"] == modelo_carro)].copy()
    if df_modelo.empty:
        return None, "Modelo não encontrado no dataset"

    # Ordenar cronologicamente
    df_modelo = df_modelo.sort_values(["year_of_reference", "month_of_reference"]).reset_index(drop=True)

    # Criar variável t (tempo)
    df_modelo["t"] = np.arange(len(df_modelo))

    X = df_modelo[["t"]]
    y = df_modelo["avg_price_brl"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Último ponto (último mês do dataset)
    t_final = df_modelo["t"].iloc[-1]

    # Previsão para o último ponto real do dataset
    y_pred = float(modelo.predict([[t_final]])[0])

    # Preço real disponível no dataset
    preco_real = float(df_modelo["avg_price_brl"].iloc[-1])

    # Métricas
    mae = abs(preco_real - y_pred)
    mse = (preco_real - y_pred) ** 2
    rmse = np.sqrt(mse)
    erro_percentual = mae / preco_real * 100

    # Monta resposta no formato que o app.py usa
    res = {
        "modelo": "Modelo 1",

        # dados do carro
        "marca": marca,
        "modelo_carro": modelo_carro,
        "ano": ano_modelo,

        # valores
        "y_pred": y_pred,
        "preco_previsto": y_pred,
        "preco_real": preco_real,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "erro_percentual": erro_percentual,

        # para gráficos:
        "t": df_modelo["t"],
        "y_real": df_modelo["avg_price_brl"],
        "modelo_obj": modelo
    }

    return res, None



# ======================================
#  FUNÇÃO 3 — GERAR GRÁFICO PARA STREAMLIT
# ======================================
def plot_regressao(res):
    t = res["t"].to_numpy().reshape(-1, 1)
    y_real = res["y_real"].to_numpy()
    y_pred = res["y_pred"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Treinar modelo interno só para plotar linha completa
    modelo_plot = LinearRegression()
    modelo_plot.fit(t, y_real)
    y_full_pred = modelo_plot.predict(t)

    ax.plot(res["t"], y_full_pred, label="Regressão Linear", linewidth=2)
    ax.scatter(res["t"], y_real, color="black", label="Preço Real")
    ax.scatter([t[-1]], [y_pred], color="red", s=90, label="Previsão Último mês")
    ax.scatter([t[-1]], [res["preco_real"]], color="green", s=90, label="Preço Real Último mês")

    ax.set_xlabel("Mês (sequência)")
    ax.set_ylabel("Preço (R$)")
    ax.set_title(f"{res['marca']} {res['modelo_carro']} — Previsão do último mês")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig
