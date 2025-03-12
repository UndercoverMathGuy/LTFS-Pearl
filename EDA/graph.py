import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('data/train_data.csv')
ref_var = 'Target_Variable/Total Income'
train_data[ref_var] = np.log(train_data[ref_var])  # apply log transformation
not_needed = ['FarmerID','State','REGION','SEX', 'CITY','Zipcode','DISTRICT','VILLAGE',	'MARITAL_STATUS','Location']

for col in train_data.columns:
    if col == ref_var or col in not_needed:
        continue
    plt.figure()
    if train_data[col].dtype == '0' or str(train_data[col].dtype) == 'category':
        sns.boxplot(data=train_data, x=col, y=ref_var)
    else:
        sns.scatterplot(data=train_data, x=col, y=ref_var)
    plt.title(f'{col} vs {ref_var}')
    plt.show()