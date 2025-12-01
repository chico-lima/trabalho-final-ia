import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==============================
# 1. Carregar dataset
# ==============================
df = pd.read_csv("/dataset/fipe_cars.csv")

print("Registros totais:", len(df))

# ==============================
# 2. Conversões e limpeza
# ==============================
map_meses = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

df["month_of_reference"] = df["month_of_reference"].map(map_meses)
df["year_of_reference"] = df["year_of_reference"].astype(int)

df = df.sort_values(["brand", "model", "year_model",
                     "year_of_reference", "month_of_reference"])

df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

# ==============================
# 3. Filtrar modelo exato
# ==============================
MARCA = "Acura"
MODELO = "Legend 3.2/3.5"
ANO_MODELO = 1998

df_modelo = df[
    (df["brand"] == MARCA) &
    (df["model"] == MODELO) &
    (df["year_model"] == ANO_MODELO)
].copy()

print(f"\nRegistros filtrados: {len(df_modelo)}")

df_modelo = df_modelo.sort_values(["year_of_reference", "month_of_reference"]).reset_index(drop=True)

# ==============================
# 4. Criar sequência temporal t
# ==============================
df_modelo["t"] = np.arange(len(df_modelo))

print(df_modelo[["ref", "avg_price_brl", "t"]])

# ==============================
# 5. Treinar modelo
# ==============================
X = df_modelo[["t", "year_model"]]    # ano_modelo constante
y = df_modelo["avg_price_brl"]

modelo = LinearRegression()
modelo.fit(X, y)

# ==============================
# 6. Previsão para FEV/2023
# ==============================
t_fev23 = df_modelo["t"].max() + 1
X_fev23 = np.array([[t_fev23, ANO_MODELO]])

y_fev23_pred = modelo.predict(X_fev23)[0]

# Definir manualmente o preço real (pois o dataset não possui fev/23)
preco_real = 27666.00

print("\n===== PREVISÃO PARA FEVEREIRO/2023 =====")
print(f"t = {t_fev23}")
print(f"Preço previsto = R$ {y_fev23_pred:.2f}")

# ==============================
# 7. Métricas de desempenho
# ==============================
mae  = mean_absolute_error([preco_real], [y_fev23_pred])
mse  = mean_squared_error([preco_real], [y_fev23_pred])
rmse = np.sqrt(mse)

# R² não é válido com apenas 1 amostra → irá retornar "nan"
r2 = r2_score([preco_real], [y_fev23_pred])

erro_percentual = abs(preco_real - y_fev23_pred) / preco_real * 100

print("\n===== MÉTRICAS DE DESEMPENHO =====")
print(f"MAE               : {mae:.2f}")
print(f"MSE               : {mse:.2f}")
print(f"RMSE              : {rmse:.2f}")
print(f"R²                : {r2}")
print(f"Erro Percentual   : {erro_percentual:.2f}%")

# ==============================
# 8. Gráfico
# ==============================
plt.figure(figsize=(12,5))

t_line = np.arange(t_fev23 + 1)
year_model_arr = np.full_like(t_line, ANO_MODELO)
X_line = np.column_stack((t_line, year_model_arr))
y_line_pred = modelo.predict(X_line)

plt.plot(t_line, y_line_pred, label="Regressão Linear")
plt.scatter(df_modelo["t"], df_modelo["avg_price_brl"], color='black', label="Valores reais")

plt.scatter(t_fev23, y_fev23_pred, color='red', s=120, label="Previsão Fev/2023 (Previsto)")
plt.scatter(t_fev23, preco_real, color='green', s=120, label="Fev/2023 (Real)")

plt.title(f"Previsão — {MARCA} {MODELO} ({ANO_MODELO})")
plt.xlabel("t (meses consecutivos)")
plt.ylabel("Preço (R$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
