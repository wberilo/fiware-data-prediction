import numpy as np
from tensorflow.keras.models import Sequential, load_model

# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# import numpy as np
# import random

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# csv_file = './caso_full.csv'

# df = pd.read_csv(csv_file)

# df.head(20)

# df['city'].nunique()

# colunas = ['city','city_ibge_code','date','estimated_population','last_available_confirmed_per_100k_inhabitants','last_available_death_rate','new_confirmed']
# df = df[colunas]

# df = df.dropna(subset=['city'])
# df = df.dropna(subset=['city_ibge_code'])

# # Converter a coluna 'data' para o tipo datetime
# df['date'] = pd.to_datetime(df['date'])

# # Ordenar o DataFrame por cidade e data
# df = df.sort_values(by=['city', 'date'])

# df['city_ibge_code'] = df['city_ibge_code'].astype(int)

# df = df.reset_index()

# df_agrupado_por_cidade = pd.DataFrame(df[['city','index']].groupby('city').count())

# df_agrupado_por_cidade['index'].plot(kind='line', figsize=(8, 4), title='dias')
# plt.gca().spines[['top', 'right']].set_visible(False)

# df.info()

# # Função para criar janelas de tempo
# def create_time_windows(df, window_size, horizon):
#     X, y = [], []
#     for i in range(len(df) - window_size - horizon + 1):
#         X.append(df['new_confirmed'].iloc[i:i + window_size].values)
#         y.append(df['new_confirmed'].iloc[i + window_size:i + window_size + horizon].values)
#     return np.array(X), np.array(y)

# # Parâmetros da janela de tempo
# window_size = 10  # Tamanho da janela histórica
# horizon = 5   # Horizonte de previsão (quantos passos no futuro queremos prever)

# all_cities = list(df['city_ibge_code'].unique())
# random.seed(809)
# cities = random.sample(all_cities, 385)

# X_list, y_list = [], []
# for idx, city in enumerate(cities):
#     city_df = df[df['city_ibge_code'] == city]
#     X_city, y_city = create_time_windows(city_df, window_size, horizon)
#     X_list.append(X_city)
#     y_list.append(y_city)
#     print(idx)

# # Concatenar os dados de todas as cidades
# X = np.concatenate(X_list, axis=0)
# y = np.concatenate(y_list, axis=0)

# # Verificar os formatos das amostras e rótulos
# print(f'Formato de X: {X.shape}')
# print(f'Formato de y: {y.shape}')

# # Dividir os dados em conjuntos de treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Verificar os formatos dos conjuntos de treino e teste
# print(f'Formato de X_train: {X_train.shape}')
# print(f'Formato de X_test: {X_test.shape}')
# print(f'Formato de y_train: {y_train.shape}')
# print(f'Formato de y_test: {y_test.shape}')

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Treinar o modelo de Regressão Linear
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # Fazer previsões
# y_pred_lr = lr.predict(X_test)

# # Avaliar o modelo
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# print(f'MSE Regressão Linear: {mse_lr}')

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # Treinar o modelo de Random Forest
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Fazer previsões
# y_pred_rf = rf.predict(X_test)

# # Avaliar o modelo
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# print(f'MSE Random Forest: {mse_rf}')

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam

# # Reformatar os dados para LSTM
# X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Construir o modelo LSTM
# model = Sequential()
# model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
# model.add(Dense(horizon))
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# # Treinar o modelo LSTM
# model.fit(X_train_lstm, y_train, epochs=30, batch_size=64, validation_data=(X_test_lstm, y_test), verbose=2)


# # Avaliar o modelo

# model.save('covid_lstm.h5')
# print("Modelo salvo com sucesso!")

# Carregar o modelo salvo
xtest = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

model_carregado = load_model('covid_lstm.h5')
print("Modelo carregado com sucesso!")

xtest_reshaped = np.reshape(xtest, (1, xtest.shape[0], 1))
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)
y_pred_carregado = model_carregado.predict(xtest_reshaped)
print(y_pred_carregado)