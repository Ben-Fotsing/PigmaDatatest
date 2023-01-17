import time
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
##
dat = pd.read_csv('D:\BenLearning\Mes Formations\diabetes.csv')
print(dat.head(10))
x = dat.drop('Outcome', axis=1)
y = dat['Outcome']
##
model= Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
##
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## 206 sec
t1 = time.time()

model.fit(x,y, epochs=150, batch_size=10,)

t2 = time.time() - t1
print('time is ',t2)

## confusion matrix
ypred = model.predict(x)
cm = confusion_matrix(np.array(y),np.rint(ypred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
