xtest = [[[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]]]
from tensorflow.keras.models import load_model

model_carregado = load_model('covid_lstm.h5')
print("Modelo carregado com sucesso!")

y_pred_carregado = model_carregado.predict(xtest)
print(y_pred_carregado)