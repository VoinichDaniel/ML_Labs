import pandas as pd

def prior(data, column_a, value_a):
    return len(data[data[column_a] == value_a]) / (len(data)+0.00001)
def likelyhood(data_only_a, column_b, value_b):
    return len(data_only_a[data_only_a[column_b] == value_b]) / (len(data_only_a)+0.00001)
def evidence(data, column_a, column_b, value_b):
    P_B = 0
    for a_i in data[column_a].unique():
        P_B += prior(data, column_a, a_i) \
* likelyhood(data[data[column_a] == a_i], column_b, value_b)
    return P_B

column_a = 'Play'
column_b = 'Weather'
data = pd.DataFrame({column_a:['no', 'no', 'no', 'yes', 'yes', 'no', 'no','yes','yes','yes', 'yes','yes','yes','yes'],
column_b:['Rainy', 'Rainy', 'Rainy', 'Rainy', 'Rainy', 'Sunny', 'Sunny','Sunny','Sunny','Sunny', 'Overcast','Overcast','Overcast','Overcast']})
B_hypothes = 'Sunny'
A_value = 'yes'
_prior = prior(data, column_a, A_value)
_likelyhood = likelyhood(data[data[column_a] == A_value], column_b, B_hypothes)
_evidence = evidence(data, column_a, column_b, B_hypothes)
#P(A_value|B_hypothes)
P = _prior*_likelyhood/_evidence
print('Вероятность: ')
print(f'P({A_value}|{B_hypothes}) =', P)
