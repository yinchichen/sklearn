import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

boston_dataset = datasets.load_boston()

X_full = boston_dataset.data
Y = boston_dataset.target

boston = pd.DataFrame(X_full)
boston.columns = boston_dataset.feature_names
boston['PRICE'] = Y
boston.head()

boston.info()

plt.style.use('seaborn-whitegrid')
plt.scatter(boston.CHAS, boston.PRICE)
plt.xlabel('CHAS')
plt.ylabel('PRICE')
plt.show()

'''
It can be seen that there is no obvious correlation between the two, 
and the following data analysis can exclude the attribute CHAS.
'''

sns.pairplot(boston, vars=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'PRICE'])

sns.pairplot(boston, vars=['DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'])

sns.pairplot(boston, vars=['CRIM', 'RM', 'LSTAT', 'PRICE'])
'''
It can be seen from the figure that PRICE has a non-linear relationship with LSTAT and CRIM, 
and a more linear relationship with RM.
'''
