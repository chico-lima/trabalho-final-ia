import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==============================
# 1. Carregar dataset
# ==============================
# Leitura do arquivo FIPE 2022
df = pd.read_csv("/dataset/fipe_2022.csv")

print("Registros totais antes da limpeza:", len(df))

# ==============================
# 2. Limpeza básica
# ==============================
# Remove colunas irrelevantes para a previsão
df = df.drop([
    "fipe_code",
    "authentication",
    "gear",
], axis=1)

# Remove registros duplicados (caso exista repetição)
df = df.drop_duplicates()

# ==============================
# 3. Converter mês (nomes → números)
# ==============================
# Mapeamento manual de nome do mês para número (necessário para ordenar)
map_meses = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# Converte textos de mês/ano para inteiros
df["month_of_reference"] = df["month_of_reference"].map(map_meses)
df["year_of_reference"] = df["year_of_reference"].astype(int)

# Cria coluna ref no formato YYYY/MM — útil para rótulos de gráfico
df["ref"] = df["year_of_reference"].astype(str) + "/" + df["month_of_reference"].astype(str).str.zfill(2)

# Ordena os dados de forma consistente
df = df.sort_values(["brand", "model", "year_of_reference", "month_of_reference"])

# ==============================
# 4. Filtrar marca e modelo desejado
# ==============================
MARCA = "Acura"
MODELO = "Legend 3.2/3.5"

# Seleciona apenas linhas do carro escolhido
df_modelo = df[(df["brand"] == MARCA) & (df["model"] == MODELO)]
print(f"Registros encontrados de {MARCA} {MODELO}: {len(df_modelo)}")

# Garante que existam meses suficientes para treinar um modelo mensal
if len(df_modelo) < 12:
    raise SystemExit("ERRO: Poucos registros para treinar previsão mensal.")

# ==============================
# 5. Filtrar apenas ano 2022
# ==============================
df_2022 = df_modelo[df_modelo["year_of_reference"] == 2022].copy()

print("Registros apenas de 2022:", len(df_2022))
print(df_2022[["ref", "avg_price_brl"]])

# ==============================
# 6. Criar variável temporal t
# ==============================
# Ordena meses e cria t = índice temporal (0 a 11)
df_2022 = df_2022.sort_values("month_of_reference").reset_index(drop=True)
df_2022["t"] = np.arange(len(df_2022))

# ==============================
# 7. Treino (Jan–Nov) e Teste (Dez)
# ==============================
# Treina usando meses 1 a 11 → prevê mês 12
df_train = df_2022[df_2022["month_of_reference"] <= 11]
df_test  = df_2022[df_2022["month_of_reference"] == 12]

# X = t (tempo) | y = preço
X_train = df_train[["t"]]
y_train = df_train["avg_price_brl"]

X_test = df_test[["t"]]
y_test = df_test["avg_price_brl"]

print("\nTreino:", len(X_train), "| Teste (dez):", len(X_test))

# ==============================
# 8. Regressão Linear
# ==============================
# Treina o modelo com relação linear entre tempo t e preço
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Gera previsão para dezembro
y_pred = modelo.predict(X_test)

# ==============================
# 9. Resultados
# ==============================
print("\n===== Previsão de DEZEMBRO para esse modelo =====")
print(f"ref={df_test['ref'].iloc[0]}  "
      f"real=R${y_test.iloc[0]:.2f}  "
      f"previsto=R${y_pred[0]:.2f}")

# ==============================
# 9A. Métricas de desempenho
# ==============================

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
erro_percentual = (abs(y_test.iloc[0] - y_pred[0]) / y_test.iloc[0]) * 100

print("\n===== MÉTRICAS DE DESEMPENHO =====")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")
print(f"Erro Percentual   : {erro_percentual:.2f}%")

# ==============================
# 10. Gráfico da Regressão
# ==============================

plt.figure(figsize=(10,5))

# Gera a linha completa da regressão (t = 0..11)
t_full = df_2022["t"].values.reshape(-1,1)
y_full_pred = modelo.predict(t_full)

# Linha da regressão
plt.plot(df_2022["ref"], y_full_pred, label="Regressão Linear", linewidth=2)

# Pontos reais do preço FIPE
plt.scatter(df_2022["ref"], df_2022["avg_price_brl"], color='black', label="Preço Real")

# Destaque de dezembro (real vs previsto)
plt.scatter(df_test["ref"], y_pred, color='red', s=100, label="Previsão Dezembro")
plt.scatter(df_test["ref"], y_test, color='green', s=100, label="Real Dezembro")

# Ajustes visuais
plt.xticks(rotation=45)
plt.xlabel("Mês de Referência (2022)")
plt.ylabel("Preço (R$)")
plt.title(f"Regressão Linear — {MARCA} {MODELO} — Previsão Dezembro")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
