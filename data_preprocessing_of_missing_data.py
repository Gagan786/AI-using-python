import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
head = pd.read_csv('C:/Users/User/Downloads/1572335298_dataset-new.csv')
x=head.iloc[:,2:-1].values
y=head.iloc[:,3].values
print(x)
print(y)

imp = Imputer(missing_values=np.nan, strategy='mean')
imp=imp.fit(x[:,:])
x[:,:]=imp.transform(x[:,:])
print(x)