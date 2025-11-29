import pandas as pd
import numpy as np
 
df = pd.read_csv("/dataset/fipe_2022.csv")
print("Arquivo carregado!")

# Remover colunas irrelevantes
df = df.drop(["fipe_code", "authentication", "gear", "year_model"], axis=1)

# Remover duplicatas
df = df.drop_duplicates()

# Mapear meses
mapeamento = {
    'January': "1", 'February': "2", 'March': "3", 'April': "4",
    'May': "5", 'June': "6", 'July': "7", 'August': "8",
    'September': "9", 'October': "10", 'November': "11", 'December': "12"
}

df["month_of_reference"] = df["month_of_reference"].map(mapeamento)

# Converter ano pra string
df["year_of_reference"] = df["year_of_reference"].astype(str)

# Criar coluna de data
df["date"] = df["month_of_reference"].str.cat(df["year_of_reference"], sep="/")
df["date"] = pd.to_datetime(df["date"], format="%m/%Y").dt.strftime('%Y/%m')

# Remover colunas antigas
df = df.drop(columns=["month_of_reference", "year_of_reference"])

print(df.columns)

df_carro_marca_marco2022 = df[
    (df["date"] == "2022/03") &
    (df["brand"] == "Acura")
]

print(df_carro_marca_marco2022)
