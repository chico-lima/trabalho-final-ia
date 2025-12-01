import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# 1. Função: carregar dataset
# ============================================
def carregar_dados(caminho="/dataset/fipe_2022.csv"):
    df = pd.read_csv(caminho)

    # Limpeza
    df = df.drop(["fipe_code", "authentication", "gear"], axis=1, errors='ignore')
    df = df.drop_duplicates()

    # Converte mês para número
    map_meses = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    df["month_of_reference"] = df["month_of_reference"].map(map_meses)
    df["year_of_reference"] = df["year_of_reference"].astype(int)
    df["year_model"] = df["year_model"].astype(int)
    df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

    return df


# ============================================
# 2. Função principal para o modelo
# ============================================
def executar_modelo(df, marca, modelo_carro, ano_modelo):

    df_modelo = df[
        (df["brand"] == marca) &
        (df["model"] == modelo_carro) &
        (df["year_model"] == ano_modelo)
    ]

    if df_modelo.empty or len(df_modelo) < 12:
        return None, "Poucos registros para este ano-modelo."

    # Filtrar apenas 2022
    df_2022 = df_modelo[df_modelo["year_of_reference"] == 2022].copy()
    df_2022 = df_2022.sort_values("month_of_reference").reset_index(drop=True)

    if len(df_2022) < 12:
        return None, "O dataset não possui os 12 meses de 2022 deste veículo."

    # Variável temporal
    df_2022["t"] = np.arange(len(df_2022))

    # Treino = meses 1–11
    df_train = df_2022[df_2022["month_of_reference"] <= 11]
    df_test  = df_2022[df_2022["month_of_reference"] == 12]

    X_train = df_train[["t"]]
    y_train = df_train["avg_price_brl"]

    X_test = df_test[["t"]]
    y_test = df_test["avg_price_brl"].values

    # Modelo Linear
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)

    # Previsão
    y_pred = modelo_lr.predict(X_test.values)

    # MÉTRICAS
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else None
    erro_percentual = (abs(y_test[0] - y_pred[0]) / y_test[0]) * 100

    # ============================================
    # 🔥 SAÍDA PADRONIZADA PARA STREAMLIT (app.py)
    # ============================================
    resultado = {
        "modelo": "Modelo 2",

        "marca": marca,
        "modelo_carro": modelo_carro,
        "ano": ano_modelo,

        # compatível com todos os modelos
        "preco_previsto": float(y_pred[0]),
        "preco_real": float(y_test[0]),
        "erro_percentual": float(erro_percentual),

        # métricas
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": r2,

        # valores necessários para o gráfico
        "t": df_2022["t"],
        "y_real": df_2022["avg_price_brl"],

        # padronização exigida pelo plot_regressao()
        "y_pred": float(y_pred[0]),
        "y_test": float(y_test[0]),

        # info extra
        "df_2022": df_2022,
        "df_test": df_test,
        "modelo_obj": modelo_lr
    }

    return resultado, None


# ============================================
# 3. Função para criar o gráfico
# ============================================
def plot_regressao(res):

    df_2022 = res["df_2022"]
    df_test = res["df_test"]

    y_test = res["y_test"]
    y_pred = res["y_pred"]

    modelo_lr = res["modelo_obj"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Linha completa com o mesmo modelo
    t_full = df_2022["t"].values.reshape(-1, 1)
    y_full_pred = modelo_lr.predict(t_full)

    ax.plot(df_2022["ref"], y_full_pred, label="Regressão Linear", linewidth=2)
    ax.scatter(df_2022["ref"], df_2022["avg_price_brl"], color="black", label="Preço Real")
    ax.scatter(df_test["ref"], [y_pred], color="red", s=90, label="Prev Dezembro")
    ax.scatter(df_test["ref"], [y_test], color="green", s=90, label="Real Dezembro")

    ax.set_xlabel("Mês (2022)")
    ax.set_ylabel("Preço (R$)")
    ax.set_title(f"{res['marca']} {res['modelo_carro']} {res['ano']} — Previsão Dezembro")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig
