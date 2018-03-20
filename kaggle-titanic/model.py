import pandas as pd
from sklearn.tree import DecisionTreeClassifier #not Regressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import Imputer #needed to fill empty values
import numpy as np

#new imports:
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# feature engineering
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).astype(int)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)


#train.to_csv('revised_input.csv', index=False)

imputer = Imputer()
columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Name_length']
train_X = imputer.fit_transform(train[columns])
test_X = imputer.fit_transform(test[columns])
train_y = train.Survived


model1 = GradientBoostingClassifier( n_estimators= 500,
     #max_features=0.2,
    max_depth= 5,
    min_samples_leaf= 2,
    verbose= 0
)
model1.fit(train_X, train_y)
model2 = AdaBoostClassifier(n_estimators= 500,
    learning_rate=0.75)
model2.fit(train_X, train_y)
model3 = RandomForestClassifier(n_jobs=-1,
     n_estimators=500,
     warm_start=True, 
     #'max_features': 0.2,
     max_depth=6,
     min_samples_leaf=2,
     max_features='sqrt',
     verbose=0

)
model3.fit(train_X, train_y)
model4 = ExtraTreesClassifier(n_jobs=-1,
    n_estimators=500,
    #max_features=0.5,
    max_depth=8,
    min_samples_leaf=2,
    verbose=0)
model4.fit(train_X, train_y)
model5 = SVC(kernel='linear',
    C=0.025)
model5.fit(train_X, train_y)

train_X1 = model1.predict(train_X)
train_X2 = model2.predict(train_X)
train_X3 = model3.predict(train_X)
train_X4 = model4.predict(train_X)
train_X5 = model5.predict(train_X)

train_X1 = train_X1[:, np.newaxis]
train_X2 = train_X2[:, np.newaxis]
train_X3 = train_X3[:, np.newaxis]
train_X4 = train_X4[:, np.newaxis]
train_X5 = train_X5[:, np.newaxis]

test_X1 = model1.predict(test_X)
test_X2 = model2.predict(test_X)
test_X3 = model3.predict(test_X)
test_X4 = model4.predict(test_X)
test_X5 = model5.predict(test_X)

test_X1 = test_X1[:, np.newaxis]
test_X2 = test_X2[:, np.newaxis]
test_X3 = test_X3[:, np.newaxis]
test_X4 = test_X4[:, np.newaxis]
test_X5 = test_X5[:, np.newaxis]

#print(train_X1.shape)

new_train_X = np.concatenate((train_X1, train_X2, train_X3, train_X4, train_X5), axis=1)
new_test_X = np.concatenate((test_X1, test_X2, test_X3, test_X4, test_X5), axis=1)

final_model = GradientBoostingClassifier(n_estimators= 500,
     #max_features=0.2,
    max_depth=5,
    min_samples_leaf=2,
    verbose=0)
final_model.fit(new_train_X, train_y)

test_y = final_model.predict(new_test_X)
test_y = pd.DataFrame(test_y, columns=['Survived'])

test_ids = test.PassengerId
prediction = pd.DataFrame(pd.concat([test_ids, test_y], axis=1), columns=['PassengerId', 'Survived'])
prediction.to_csv('prediction.csv', index=False)
