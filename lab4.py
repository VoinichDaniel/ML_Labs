import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv('customer_shopping_cart_data.csv')
k_max = 10

data = data.drop(['Region'], axis=1)
if 'Channel' in data.columns:
    data = data.drop(['Channel'], axis=1)

data.isnull().sum()
sse = []
silhouette_coefficients = []

for k in range(2, k_max + 1):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)
    silhouette_coefficients.append(silhouette_score(data, kmeans.predict(data)))

plt.plot(range(2, k_max + 1), sse, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Суммарное квадратичное отклонение')
plt.show()

plt.plot(range(2, k_max + 1), silhouette_coefficients, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Коэффициент силуэта')
plt.show()