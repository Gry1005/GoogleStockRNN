import pandas
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential

#读取数据
googl=pandas.read_csv('../data/google_stock_prics.csv')
#print(googl.head())

#一共2156行数据，最后300行作为测试集
training_set=googl.iloc[0:2156-300,4:5]
test_set=googl.iloc[2156-300:,4:5]
#print(training_set.shape)

#标准化

sc=MinMaxScaler(feature_range=(0,1))
training_set=sc.fit_transform(training_set)
test_set=sc.fit_transform(test_set)

#用每60步，去预测第61步
x_train=[]
y_train=[]

for i in range(60,len(training_set)):
    x_train.append(training_set[i-60:i])
    y_train.append(training_set[i])

x_train, y_train = np.array(x_train),np.array(y_train)

#print(x_train.shape)
#print(y_train.shape)

#改变x_train的结构，最里面的数组，每个元素要是一个单独的小数组
#x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#print(x_train.shape)

x_test=[]
y_test=[]

for i in range(60,len(test_set)):
    x_test.append(test_set[i-60:i])
    y_test.append(test_set[i])

x_test, y_test = np.array(x_test),np.array(y_test)

#print(x_test.shape)
#print(y_test.shape)


#网络结构
gNet=Sequential()

gNet.add(LSTM(
    units=50,
    return_sequences=True, #True: 返回全部输入的hidden state, False:只返回一个hidden state
    input_shape=(x_train.shape[1],1) #time_steps, 每一步的输入个数
))
gNet.add(Dropout(0.2))

gNet.add(LSTM(
    units=50,
    return_sequences=True
))
gNet.add(Dropout(0.2))

gNet.add(LSTM(
    units=50,
    return_sequences=True
))
gNet.add(Dropout(0.2))

gNet.add(LSTM(
    units=50,
    return_sequences=False
))
gNet.add(Dropout(0.2))

gNet.add(Dense(units=1))

#编译训练
gNet.compile(optimizer='adam',loss='mean_squared_error')

print('Training-------------')
gNet.fit(x_train,y_train,epochs=100,batch_size=100)

#测试结果
print('Testing-------------')
loss=gNet.evaluate(x_test,y_test)


#作图
predicted_stock_price=gNet.predict(x_test)

#对数据的规模进行还原
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(y_test)

import matplotlib.pyplot as plt
plt.plot(real_stock_price     , color = 'red' , label = 'Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()








