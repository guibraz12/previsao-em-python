import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

temperaturas = np.array([22, 24, 23, 25, 26, 27, 28])

dias = np.array([1, 2, 3, 4, 5, 6, 7])

df = pd.DataFrame({'dias': dias, 'temperaturas': temperaturas})

modelo = LinearRegression()
modelo.fit(df[['dias']], df['temperaturas'])

dia_seguinte = pd.DataFrame({'dias': [8]})
previsão = modelo.predict(dia_seguinte)

print("Previsão da temperatura para o dia seguinte:", previsão[0])

plt.scatter(dias, temperaturas, color="b")
plt.plot(df[['dias']], modelo.predict(df[['dias']]), color='g', label='Linha de Tendência')
plt.scatter(dia_seguinte, previsão, color="r", label='Previsão')
plt.xlabel('Dias')
plt.ylabel('Temperatura')
plt.legend()
plt.show()