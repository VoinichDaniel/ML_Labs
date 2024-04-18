import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def task():
    x = np.array([451, 163, 513, 112, 123, 561, 312, 123, 456, 156, 126, 131])
    y = np.array([145, 123, 235, 145, 120, 160, 100, 240, 135, 123, 230, 125])

    xy = x * y
    x_squared = x ** 2
    y_squared = y ** 2
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    x_diff_squared = x_diff ** 2
    y_diff_squared = y_diff ** 2
    xy_sum = np.sum(xy)
    x_squared_sum = np.sum(x_squared)
    y_squared_sum = np.sum(y_squared)
    x_diff_squared_sum = np.sum(x_diff_squared)
    y_diff_squared_sum = np.sum(y_diff_squared)

    print("х * y: ", xy)
    print("Квадрат x: ", x_squared)
    print("Квадрат y: ", y_squared)
    print("Отклонение от среднего значения x: ", x_diff)
    print("Отклонение от среднего значения y: ", y_diff)
    print("Отклонение от среднего значения x в квадрате: ", x_diff_squared)
    print("Отклонение от среднего значения y в квадрате: ", y_diff_squared)
    print("Сумма (x * y): ", xy_sum)
    print("Сумма квадратов x: ", x_squared_sum)
    print("Сумма квадратов y: ", y_squared_sum)
    print("Сумма квадратов отклонений x: ", x_diff_squared_sum)
    print("Сумма квадратов отклонений y: ", y_diff_squared_sum)

    coefficients = np.polyfit(x, y, 1)

    print("\nУравнение регрессии: y = {:.2f}x + {:.2f}".format(coefficients[0], coefficients[1]))

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    std_err = model.bse

    print("\nСтандартные ошибки коэффициентов: ")
    print(std_err)

    def regression_equation(x):
        return 0.02 * x + 150.72

    x_values = np.array([451,163,513,112,123,561,312,123,456,156,126,131])
    y_pred = regression_equation(x_values)

    plt.scatter(x, y)
    plt.title('Корреляционное поле')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.scatter(x, y)
    plt.plot(x, y_pred, color='orange')
    plt.title('Регрессионная линия')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


task()