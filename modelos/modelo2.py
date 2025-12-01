import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Carregar dataset
# ==============================
# Carrega o dataset FIPE completo de 2022
df = pd.read_csv("/dataset/fipe_2022.csv")

print("Registros totais antes da limpeza:", len(df))

# ==============================
# 2. Limpeza básica
# ==============================
# Remove colunas que não serão utilizadas no modelo
df = df.drop([
    "fipe_code",
    "authentication",
    "gear",
], axis=1)

# Remove duplicidades caso existam
df = df.drop_duplicates()

# ==============================
# 3. Converter mês (nomes → números)
# ==============================
# Mapeamento texto → número
map_meses = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Converte colunas para formatos apropriados
df["month_of_reference"] = df["month_of_reference"].map(map_meses)
df["year_of_reference"] = df["year_of_reference"].astype(int)
df["year_model"] = df["year_model"].astype(int)

# Cria coluna ref no formato YYYY/MM (para visualização)
df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

# ==============================
# 4. Filtrar marca, modelo E ANO DO MODELO
# ==============================
# Aqui você fixa o ano de fabricação específico,
# diferente do modelo anterior que misturava vários anos-modelo.

MARCA = "Acura"
MODELO = "Legend 3.2/3.5"
ANO_MODELO = 1998   # <- ano do modelo usado como filtro

# Filtra primeiro marca e modelo,
# depois restringe para um único ano de fabricação
df_modelo = df[
    (df["brand"] == MARCA) &
    (df["model"] == MODELO) &
    (df["year_model"] == ANO_MODELO)
]

print(f"Registros encontrados de {MARCA} {MODELO} {ANO_MODELO}: {len(df_modelo)}")

# Garante que existem registros suficientes para prever dezembro
if len(df_modelo) < 12:
    raise SystemExit("ERRO: Poucos registros para esse ano-modelo.")

# ==============================
# 5. Filtrar apenas ano de referência 2022
# ==============================
# Agora pegamos apenas os meses de 2022 desse carro
df_2022 = df_modelo[df_modelo["year_of_reference"] == 2022].copy()

print("Registros apenas de 2022:", len(df_2022))
print(df_2022[["ref", "avg_price_brl"]])

# ==============================
# 6. Criar variável temporal t
# ==============================
# Ordena meses e cria t = 0, 1, ..., 11 → usado pelo modelo linear
df_2022 = df_2022.sort_values("month_of_reference").reset_index(drop=True)
df_2022["t"] = np.arange(len(df_2022))

# ==============================
# 7. Treino = Jan–Nov / Teste = Dez
# ==============================
# Mantém a mesma ideia do modelo 1:
# treinamos com meses 1–11 e testamos com dezembro

df_train = df_2022[df_2022["month_of_reference"] <= 11]
df_test  = df_2022[df_2022["month_of_reference"] == 12]

# X (entrada) → apenas o tempo t
X_train = df_train[["t"]]
y_train = df_train["avg_price_brl"]

X_test = df_test[["t"]]

preco_real = 27666.00  # Preco real de FEVEREIRO 
 
print("\nTreino:", len(X_train), "| Teste (dez):", len(X_test))

# Como estamos usando preco_real manual, recriamos y_test corretamente
y_test = np.array([preco_real])

# ==============================
# 8. Regressão Linear
# ==============================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsão usando X_test (garante compatibilidade)
y_pred = modelo.predict(X_test.values)

# ==============================
# 9. Resultados
# ==============================
print("\n===== Previsão de DEZEMBRO =====")
print(f"{MARCA} {MODELO} {ANO_MODELO}")
print(f"ref={df_test['ref'].iloc[0]}  real=R${preco_real:.2f}  previsto=R${y_pred[0]:.2f}")

# ==============================
# 10. Métricas de desempenho
# ==============================
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# R² só é válido quando há mais de 1 valor no conjunto de teste
if len(y_test) > 1:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = "N/A"

erro_percentual = (abs(preco_real - y_pred[0]) / preco_real) * 100

print("\n===== MÉTRICAS DE DESEMPENHO =====")
print(f"MAE               : {mae:.2f}")
print(f"MSE               : {mse:.2f}")
print(f"RMSE              : {rmse:.2f}")
print(f"R²                : {r2}")
print(f"Erro Percentual   : {erro_percentual:.2f}%")
  
# ==============================
# 10. Gráfico
# ==============================
plt.figure(figsize=(10,5))

# Gera a linha completa da regressão (t = 0..11)
t_full = df_2022["t"].values.reshape(-1,1)
y_full_pred = modelo.predict(t_full)

# Linha da regressão
plt.plot(df_2022["ref"], y_full_pred, label="Regressão Linear", linewidth=2)

# Pontos reais
plt.scatter(df_2022["ref"], df_2022["avg_price_brl"], color='black', label="Preço Real")

# Destaque mês previsto vs real
plt.scatter(df_test["ref"], y_pred, color='red', s=80, label="Prev Dezembro")
plt.scatter(df_test["ref"], y_test, color='green', s=80, label="Real Dezembro")

# Configurações visuais
plt.xticks(rotation=45)
plt.xlabel("Mês (2022)")
plt.ylabel("Preço (R$)")
plt.title(f"{MARCA} {MODELO} {ANO_MODELO} — Previsão Dezembro")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
