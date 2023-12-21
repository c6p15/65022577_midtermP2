from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

# Preprocess
File_path = 'D:/65022577/data/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path + File_name)
df.drop(columns=('User ID'), inplace=True )

encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[:, 0:3]
y = df['Purchased']

#train test 
x_train,_,_,_ = train_test_split(x,y)
display(x_train)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

score = model.score(x,y)

print('Accuracy : ', '{:.2f}'.format(score))

# decision tree image
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names= Data_class,
              label='all',
              impurity= True,
              precision = 3,
              filled= True,
              rounded= True,
              fontsize= 16)

plt.show()

# feature importance
import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary',] 
 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)
