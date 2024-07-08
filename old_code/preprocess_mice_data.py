import pandas as pd

# Read in the data
data = pd.read_excel('data/mice_protein.xls', sheet_name='Hoja1', header=0, index_col=0)

# Impute missing values using column means
for column in data.columns[data.isnull().any()]:
    data[column].fillna(data[column].mean(), inplace=True)

# Factorize class
data['class'], _ = pd.factorize(data['class'])

data.to_csv('data/preprocessed_mice.csv')