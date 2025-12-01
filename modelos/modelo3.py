import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================
# 1. Carregar dataset
# ==============================
# Dataset maior contendo dados de 2021 até janeiro de 2023
df = pd.read_csv("/dataset/fipe_cars.csv")

print("Registros totais:", len(df))

# ==============================
# 2. Conversões e limpeza
# ==============================
# Converte nome dos meses para número (para ordenar corretamente)
map_meses = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

df["month_of_reference"] = df["month_of_reference"].map(map_meses)
df["year_of_reference"] = df["year_of_reference"].astype(int)

# Ordena para garantir consistência temporal no dataset completo
df = df.sort_values(["brand", "model", "year_model",
                     "year_of_reference", "month_of_reference"])

# Cria coluna útil para visualização YYYY/MM
df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

# ==============================
# 3. Filtrar modelo exato
# ==============================
# Carro específico que queremos prever
MARCA = "Acura"
MODELO = "Legend 3.2/3.5"
ANO_MODELO = 1998

# Mantém apenas os registros desse carro/ano-modelo
df_modelo = df[
    (df["brand"] == MARCA) &
    (df["model"] == MODELO) &
    (df["year_model"] == ANO_MODELO)
].copy()

print(f"\nRegistros filtrados: {len(df_modelo)}")

if len(df_modelo) < 15:
    print("⚠ Poucos dados, mas continuando mesmo assim...")

# ==============================
# 4. Criar sequência temporal t
# ==============================
# Ordena meses reais do carro e cria t = 0, 1, 2, ..., último mês disponível
df_modelo = df_modelo.sort_values(["year_of_reference", "month_of_reference"]).reset_index(drop=True)

df_modelo["t"] = np.arange(len(df_modelo))  # sequência temporal contínua

# Exibe tabela final usada no treinamento
print(df_modelo[["ref", "avg_price_brl", "t"]])

# ==============================
# 5. Treinar modelo com TODOS os dados reais
# ==============================
# Aqui o modelo usa:
#   X = [t, year_model]
# year_model é constante (1998), então age como um "offset"
X = df_modelo[["t", "year_model"]]
y = df_modelo["avg_price_brl"]

# Treina regressão linear
modelo = LinearRegression()
modelo.fit(X, y)

# ==============================
# 6. Criar entrada para previsão de fevereiro/2023
# ==============================
# Fevereiro/2023 é exatamente o mês seguinte ao último mês real
t_fev23 = df_modelo["t"].max() + 1

# Monta entrada com t aumentado e ano_modelo igual
X_fev23 = np.array([[t_fev23, ANO_MODELO]])

# Faz a previsão
y_fev23_pred = modelo.predict(X_fev23)[0]

print("\n===== PREVISÃO PARA FEVEREIRO/2023 =====")
print(f"t = {t_fev23}")
print(f"Preço previsto = R$ {y_fev23_pred:.2f}")

# ==============================
# 7. Gráfico completo da série temporal
# ==============================
plt.figure(figsize=(12,5))

# Gera linha de previsão contínua até fev/2023
t_line = np.arange(t_fev23 + 1)
year_model_arr = np.full_like(t_line, ANO_MODELO)

# Monta matriz de entrada completa para prever a curva inteira
X_line = np.column_stack((t_line, year_model_arr))

# Previsão completa da linha de regressão
y_line_pred = modelo.predict(X_line)

# Linha da regressão
plt.plot(t_line, y_line_pred, label="Regressão Linear")

# Pontos reais da FIPE
plt.scatter(df_modelo["t"], df_modelo["avg_price_brl"], color='black', label="Valores reais")

# Ponto estimado para fevereiro de 2023
plt.scatter(t_fev23, y_fev23_pred, color='red', s=120, label="Previsão Fev/2023")

plt.title(f"Previsão — {MARCA} {MODELO} ({ANO_MODELO})")
plt.xlabel("t (meses consecutivos)")
plt.ylabel("Preço (R$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
