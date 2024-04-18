import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset = pd.read_csv('btc.csv', index_col='Date', parse_dates=True)
data = dataset.dropna()
train = data["2018":"2022"]
test = data["2023":]

model = ExponentialSmoothing(train['Open'], trend='add', seasonal='add', seasonal_periods=365)
model_fit = model.fit()
forecast = model_fit.forecast(len(test))

print(test.index)
plt.plot(train.index, train['Open'], label='Фактические значения')
plt.plot(test.index, forecast, label='Прогноз')
plt.legend(loc='upper left')
plt.show()

rmse = root_mean_squared_error(test['Open'], forecast)
print('RMSE:', rmse)

# for row in test.iterrows():
#      print(row, "\n")