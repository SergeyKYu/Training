import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
titanic_data = pd.read_csv('train.csv', index_col='PassengerId')
titanic_test = pd.read_csv('test.csv', index_col='PassengerId')


#print(titanic_data.isnull().sum())
X = titanic_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
Y_train = titanic_data.Survived
X = pd.get_dummies(X)
X_train = X.fillna({'Age': X.Age.median()})

X_p = titanic_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)#
X_p = pd.get_dummies(X_p)
X_pred = X_p.fillna({'Age': X_p.Age.median(), 'Fare': X_p.Fare.median()})

print(X_pred.isna().sum())
X_pred_np = X_pred.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
print(Y_train.head())
#print(Y_test.isna().sum())

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
#print(X_train_np)
y_train_cat = keras.utils.to_categorical(Y_train, 2)
y_test_cat = keras.utils.to_categorical(Y_test, 2)
print(y_train_cat)

model = keras.Sequential([
    Input(shape=(10,)),
    Dense(60, activation='relu'),
    # Dropout(0.2),
    #BatchNormalization(),
    Dense(2, activation='softmax')
])
print(model.summary())
model.compile(optimizer=keras.optimizers.Adam(0.001),#'adam',#!
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_np, y_train_cat, batch_size=20, epochs=5, validation_split=0.1)
model.evaluate(X_test_np, y_test_cat)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid(True)
plt.show()

resx = model.predict(X_pred_np)
res = np.argmax(resx, axis=1)


y_pred = pd.DataFrame({'Survived': res})
y_pred.index = X_pred.index
print(y_pred.head())
y_pred.to_csv('Submission.csv')