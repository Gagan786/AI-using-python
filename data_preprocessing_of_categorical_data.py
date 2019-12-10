import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler


data=pd.read_csv("C:/Users/User/Downloads/machine_learning-master/DataPreprocessing.csv")
x=data.iloc[:,:3]
y=data.iloc[:,-1]
print(x)
print(y)
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3]) 


labelEncoder_x=LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelEncoder_y=LabelEncoder()
y=labelEncoder.fit_tranform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
sc_x=Standard_scaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
    

