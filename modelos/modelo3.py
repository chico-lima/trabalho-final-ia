import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# 1. Carregar dataset
# ============================================
def carregar_dados(caminho="/dataset/fipe_cars.csv"):
    df = pd.read_csv(caminho)
    map_meses = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["month_of_reference"] = df["month_of_reference"].map(map_meses)
    df["year_of_reference"] = df["year_of_reference"].astype(int)
    df["year_model"] = df["year_model"].astype(int)
    df = df.sort_values(["brand", "model", "year_model", "year_of_reference", "month_of_reference"])
    df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)
    return df

# ============================================
# 2. Executar modelo
# ============================================
def executar_modelo(df, marca, modelo, ano_modelo, preco_real=27666.00):
    df_modelo = df[
        (df["brand"] == marca) &
        (df["model"] == modelo) &
        (df["year_model"] == ano_modelo)
    ].copy()
    
    if len(df_modelo) < 1:
        return None, "Poucos registros para este ano-modelo."

    df_modelo = df_modelo.sort_values(["year_of_reference", "month_of_reference"]).reset_index(drop=True)
    df_modelo["t"] = np.arange(len(df_modelo))

    X = df_modelo[["t", "year_model"]]
    y = df_modelo["avg_price_brl"]

    modelo_lr = LinearRegression()
    modelo_lr.fit(X, y)

    t_fev23 = df_modelo["t"].max() + 1
    X_fev23 = np.array([[t_fev23, ano_modelo]])
    y_fev23_pred = modelo_lr.predict(X_fev23)[0]

    # Métricas
    mae  = mean_absolute_error([preco_real], [y_fev23_pred])
    mse  = mean_squared_error([preco_real], [y_fev23_pred])
    rmse = np.sqrt(mse)
    r2 = r2_score([preco_real], [y_fev23_pred])
    erro_percentual = abs(preco_real - y_fev23_pred) / preco_real * 100

    resultado = {
        "df_modelo": df_modelo,
        "t_fev23": t_fev23,
        "y_fev23_pred": y_fev23_pred,
        "preco_previsto": y_fev23_pred,
        "preco_real": preco_real,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "erro_percentual": erro_percentual,
        "marca": marca,
        "modelo": modelo,
        "ano": ano_modelo
    }
    return resultado, None

# ============================================
# 3. Gráfico
# ============================================
def plot_regressao(res):
    df_modelo = res["df_modelo"]
    t_line = np.arange(res["t_fev23"] + 1)
    year_model_arr = np.full_like(t_line, res["ano"])
    X_line = np.column_stack((t_line, year_model_arr))

    modelo_lr = LinearRegression()
    modelo_lr.fit(df_modelo[["t","year_model"]], df_modelo["avg_price_brl"])
    y_line_pred = modelo_lr.predict(X_line)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(t_line, y_line_pred, label="Regressão Linear")
    ax.scatter(df_modelo["t"], df_modelo["avg_price_brl"], color="black", label="Valores reais")
    ax.scatter(res["t_fev23"], res["y_fev23_pred"], color="red", s=120, label="Previsão Fev/2023 (Previsto)")
    ax.scatter(res["t_fev23"], res["preco_real"], color="green", s=120, label="Fev/2023 (Real)")
    ax.set_xlabel("t (meses consecutivos)")
    ax.set_ylabel("Preço (R$)")
    ax.set_title(f"{res['marca']} {res['modelo']} ({res['ano']}) — Previsão Fev/2023")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
