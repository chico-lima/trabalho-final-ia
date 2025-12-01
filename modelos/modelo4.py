import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ==============================
# 1. Carregar dataset
# ==============================
# Arquivo contém todas as cotações da FIPE com vários modelos/anos
df = pd.read_csv("/dataset/fipe_cars.csv")

print("Registros totais:", len(df))

# ==============================
# 2. Conversões e limpeza
# ==============================
# Mapeamento de meses em inglês → número
map_meses = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

df["month_of_reference"] = df["month_of_reference"].map(map_meses)
df["year_of_reference"] = df["year_of_reference"].astype(int)

# Ordenação para facilitar a criação de linhas temporais
df = df.sort_values(["brand", "model", "year_model",
                     "year_of_reference", "month_of_reference"])

# Criar ref YYYY/MM
df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

# ==============================
# 3. Filtrar modelo exato ORIGINAL
# ==============================
MARCA = "Acura"
MODELO = "Legend 3.2/3.5"
ANO_MODELO = 1998

# Função auxiliar que verifica se existe um ano-modelo específico no dataset
def existe_ano_modelo(ano):
    return df[(df["brand"] == MARCA) &
              (df["model"] == MODELO) &
              (df["year_model"] == ano)].shape[0] > 0

# ==============================
# Procurar anos vizinhos (até 2)
# Isso ajuda quando o ano-modelo alvo tem poucos registros
# Preferência: anos anteriores → depois posteriores
# ==============================
anos_vizinhos = []

# Anos anteriores
for delta in [1, 2]:
    ano = ANO_MODELO - delta
    if existe_ano_modelo(ano):
        anos_vizinhos.append(ano)
    if len(anos_vizinhos) == 2:  # já temos 2
        break

# Se ainda faltar, procurar anos acima
if len(anos_vizinhos) < 2:
    for delta in [1, 2]:
        ano = ANO_MODELO + delta
        if existe_ano_modelo(ano):
            anos_vizinhos.append(ano)
        if len(anos_vizinhos) == 2:
            break

print("\nAnos vizinhos encontrados:", anos_vizinhos)

# ==============================
# 4. Construir dataset expandido
# Junta o ano original + anos vizinhos para gerar mais dados
# Melhora muito a estabilidade da regressão
# ==============================
anos_para_buscar = [ANO_MODELO] + anos_vizinhos

df_modelo_expandido = df[
    (df["brand"] == MARCA) &
    (df["model"] == MODELO) &
    (df["year_model"].isin(anos_para_buscar))
].copy()

print(f"\nRegistros filtrados (com vizinhos): {len(df_modelo_expandido)}")

# Ordenar por data real
df_modelo_expandido = df_modelo_expandido.sort_values(
    ["year_of_reference", "month_of_reference"]
).reset_index(drop=True)

# Criar variável temporal t = 0, 1, 2, ..., N
df_modelo_expandido["t"] = np.arange(len(df_modelo_expandido))

print(df_modelo_expandido[["year_model", "ref", "avg_price_brl", "t"]])

# ==============================
# 5. Treinar modelo
# Entrada: tempo (t) + ano_modelo
# O ano_modelo permite que carros mais novos tenham valores diferentes
# ==============================
X = df_modelo_expandido[["t", "year_model"]]
y = df_modelo_expandido["avg_price_brl"]

modelo = LinearRegression()
modelo.fit(X, y)

# ==============================
# 6. Previsão (Fevereiro/2023)
# Próximo mês = último t + 1
# ==============================
t_fev23 = df_modelo_expandido["t"].max() + 1

# O ano-modelo permanece o original (1998)
X_fev23 = np.array([[t_fev23, ANO_MODELO]])

y_fev23_pred = modelo.predict(X_fev23)[0]

print("\n===== PREVISÃO PARA FEVEREIRO/2023 =====")
print(f"t = {t_fev23}")
print(f"Preço previsto = R$ {y_fev23_pred:.2f}")

# ==============================
# 6.1 — MÉTRICAS DE DESEMPENHO
# ==============================

preco_real = 27666.00  

mae  = mean_absolute_error([preco_real], [y_fev23_pred])
mse  = mean_squared_error([preco_real], [y_fev23_pred])
rmse = np.sqrt(mse)
erro_percentual = abs(preco_real - y_fev23_pred) / preco_real * 100

print("\n===== MÉTRICAS DE DESEMPENHO =====")
print(f"MAE              : {mae:.2f}")
print(f"MSE              : {mse:.2f}")
print(f"RMSE             : {rmse:.2f}")
print(f"Erro Percentual  : {erro_percentual:.2f}%")

# R² não funciona com 1 amostra → evitar erro
print("R²               : N/A (exige 2+ valores reais)")


# ==============================
# 7. Gráfico
# ==============================
plt.figure(figsize=(12,5))

# Criar linha desde t=0 até o t previsto
t_line = np.arange(t_fev23 + 1)
year_model_arr = np.full_like(t_line, ANO_MODELO)

# Matriz completa da regressão
X_line = np.column_stack((t_line, year_model_arr))

# Previsão completa da curva
y_line_pred = modelo.predict(X_line)

# Linha da regressão
plt.plot(t_line, y_line_pred, label="Regressão Linear")

# Pontos reais usados no treino
plt.scatter(df_modelo_expandido["t"], df_modelo_expandido["avg_price_brl"],
            color='black', label="Valores reais")

# Ponto previsto para fev/23
plt.scatter(t_fev23, y_fev23_pred, color='red', s=120,
            label="Previsão Fev/2023")

plt.title(f"Previsão — {MARCA} {MODELO} ({ANO_MODELO})")
plt.xlabel("t (meses consecutivos)")
plt.ylabel("Preço (R$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
