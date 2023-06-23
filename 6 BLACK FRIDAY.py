import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

svc = RandomForestRegressor()

data = pd.read_csv('D:/internship/Train.csv')
data = data.drop(["User_ID","Product_ID"],axis=1)

data['Age']=(data['Age'].str.strip('+'))
data['Stay_In_Current_City_Years']=(data['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

data['Product_Category_2'] =data['Product_Category_2'].fillna(0).astype('int64')
data['Product_Category_3'] =data['Product_Category_3'].fillna(0).astype('int64')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Age'] = le.fit_transform(data['Age'])
data['City_Category'] = le.fit_transform(data['City_Category'])

x = data.drop(['Purchase'], axis=1)
y = data['Purchase']


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)


print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test, y_pred))