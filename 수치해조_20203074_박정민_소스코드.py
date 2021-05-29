#수치해조_20203074_박정민
#케라스를 이용한 선형회귀 분석
print("<Keras 패키지>\n")
#https://skettee.github.io/post/linear_regression/#%EC%BC%80%EB%9D%BC%EC%8A%A4-keras-%EB%A1%9C-%EB%AA%A8%EB%8D%B8%EB%A7%81-modeling
import numpy as np
import pandas as pd


df = pd.read_csv("https://raw.githubusercontent.com/kmujm/NumericalAnalysis/main/dataset.csv")
df.head()


subscriber = df['Subscriber']/10000
income = df['Income']/10000
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(subscriber, income, test_size=0.2, random_state=123)

%matplotlib inline

import matplotlib.pyplot as plt

plt.title("< Scattered Data >")
plt.scatter(subscriber, income)
plt.xlabel('Subscriber (10000)')
plt.ylabel('Income (10000)')
plt.show()



from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d

x = np.array(x_train).reshape(len(x_train),1)
y = np.array(y_train).reshape(len(y_train),1)



# 손실 함수 시각화
# W,b의 범위를 결정한다.
w = np.arange(0, 700, 10)
b = np.arange(-10, 10, 0.1)
j_array = []

W, B = np.meshgrid(w, b)

# w, b를 하나씩 대응한다.
for we, be in zip(np.ravel(W), np.ravel(B)):
    y_hat = np.add(np.multiply(we, x), be)
    # Loss function
    mse = mean_squared_error(y_hat, y) 
    j_array.append(mse)

# 손실(Loss)을 구한다.

J = np.array(j_array).reshape(W.shape)

# 서피스 그래프를 그린다.
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(W, B, J, color='b', alpha=0.5)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('J')
ax.set_zticks([])
plt.title("< Loss Function >")
plt.show()


# 데이터 정규화
from sklearn import preprocessing

mm_scaler = preprocessing.MinMaxScaler()
X_train = mm_scaler.fit_transform(x)
Y_train = mm_scaler.transform(y)

plt.scatter(X_train, Y_train)
plt.title("< Scaled Data >")
plt.xlabel('Subscriber')
plt.ylabel('Income')
plt.show()

# 케라스를 이용한 모델링
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 모델을 준비한다.
model = Sequential()

# 입력 변수의 개수가 1이고 출력 개수가 1인 y=wx+b 를 생성한다.
model.add(Dense(1, input_dim=1))

# Loss funtion과 Optimizer를 선택한다.
model.compile(loss='mean_squared_error', optimizer='sgd') 

# epochs만큼 반복해서 손실값이 최저가 되도록 모델을 훈련한다.
hist = model.fit(X_train, Y_train, epochs=2000, verbose=0) 

plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# w, b 값 확인
w, b = model.get_weights()
w =  w[0][0]
b = b[0]
print('w: ', w)
print('b: ', b)

#그래프로 확인
x_scale = mm_scaler.transform(x)
y_scale = mm_scaler.transform(y)
plt.title("Linear Fit")
plt.scatter(x_scale, y_scale)
plt.plot(x_scale, w*x_scale+b, 'r')
plt.xlabel('scaled-Subscriber')
plt.ylabel('scaled-Income')
plt.show()

#성능평가.sklearn를 이용하여 R^2의 방식으로 예측률 분석.
#성능평가.sklearn를 이용하여 R^2의 방식으로 예측률 분석.
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
print("R²: ", r2_score(y_test, y_predict))

print("***********************************************************************")

#텐서플로우를 이용한 선형회귀
print("<Tensorflow 패키지>\n")
#https://tensorflow.blog/2-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D-first-contact-with-tensorflow/
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


df = pd.read_csv("https://raw.githubusercontent.com/kmujm/NumericalAnalysis/main/dataset.csv")
df.head()

#x, y데이터 분류
from sklearn.model_selection import train_test_split
x_data = df['Subscriber']/10000
y_data = df['Income']/10000
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.title("< Scattered Data >")
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_train + b

#손실함수 최솟값 구하기
loss = tf.reduce_mean(tf.square(y - y_train))


#학습률(0.0000004)에 따라 최적화 훈련.
#처음에 학습률을 0.5에서 시작했더니 발산, 0.000000001은 차이가 미세하여 최적해를 구하는데 오래 걸림.
#trial and error를 통해 적절한 학습률을 찾음.
optimizer = tf.train.GradientDescentOptimizer(0.0000004)
train = optimizer.minimize(loss)

#텐서플로에 사용되는 변수 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(20):
     sess.run(train)
     print(step, sess.run(W), sess.run(b))
     print(step, sess.run(loss))

     #최적해를 찾아가는 과정 시각화
     plt.plot(x_train, y_train, 'ro')
     plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
     plt.xlabel('Subscriber')
     plt.xlim(-300,6000)
     plt.ylim(-100000,2000000)
     plt.ylabel('Income')
plt.title("< Linear Fit >")
plt.show()

#성능평가.sklearn를 이용하여 R^2의 방식으로 예측률 분석.
from sklearn.metrics import r2_score
y_predict = []
for x in x_test:
  cx = tf.constant(x)
  y_predict.append(cx*W+b)
print("R²: ", r2_score(y_test, sess.run(y_predict)))

print("***********************************************************************")

#파이토치를 이용한 선형회귀
print("<Pytorch 패키지>\n")
#https://wikidocs.net/53560

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

torch.manual_seed(1)

df = pd.read_csv("https://raw.githubusercontent.com/kmujm/NumericalAnalysis/main/dataset.csv")
df.head()

#x, y데이터 분류
from sklearn.model_selection import train_test_split
x_data = df['Subscriber']/10000
y_data = df['Income']/10000
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)
#훈련 데이터셋 구성
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
#모델 초기화
W = torch.zeros(1, requires_grad=True) 
b = torch.zeros(1, requires_grad=True)
#optimizer 설정
optimizer = optim.SGD([W, b], lr=0.0000004)

nb_epochs = 25 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()   # gradient를 0으로 초기화
    cost.backward()   # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 업데이트

    print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), b.item(), cost.item()
    ))

#성능평가.sklearn를 이용하여 R^2의 방식으로 예측률 분석.
from sklearn.metrics import r2_score
y_predict = []
for x in x_test:
  y_predict.append(x*W+b)
print("R²: ", r2_score(y_test, y_predict))

print("***********************************************************************")

#사이킷런을 이용한 선형회귀
print("<scikit-learn 패키지>\n")
#https://www.skyer9.pe.kr/wordpress/?p=437
#import numpy as np  <--이 코드에서는 numpy가 안 쓰임
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/kmujm/NumericalAnalysis/main/dataset.csv")
df.head()


#x, y데이터 분류
from sklearn.model_selection import train_test_split
x_data = df['Subscriber']/10000
y_data = df['Income']/10000
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)

X = x_train
y = y_train
line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)


#print(line_fitter.predict([[1000]]))
print("W: ",line_fitter.coef_)
print("b: ",line_fitter.intercept_)

plt.plot(X, y, 'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
plt.title("< Linear Fit >")
plt.show()
#성능평가.sklearn를 이용하여 R^2의 방식으로 예측률 분석.
from sklearn.metrics import r2_score
y_predict = []
for x in x_test:
  y_predict.append(x*line_fitter.coef_ + line_fitter.intercept_)
print("R²: ", r2_score(y_test, y_predict))
