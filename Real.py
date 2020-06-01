print("Cloud Counselage IP")
print("Data Science Project")
print("Project name: Real/Fake News Classifier")
print("Project by:   PRASFUR TIWARI")

print("\nPROBLEM STATEMENT")

print("To build a model to accurately classify a piece of news as REAL or FAKE.\nUsing sklearn,  build a TfidfVectorizer on the provided dataset.")
print("Then, initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.")
print("On completion, create a GitHub account and create a repository. Commit your python code inside the newly created repository.")

print("\nLoading Modules...")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("\nModules Loaded!")

print("\nData Gathering/Collection")

print("\nReading Dataset")

df=pd.read_csv('https://raw.githubusercontent.com/Prasfur/Data/master/news.csv')
print(df.head())

print("Data Wrangling")

df.set_index('Unnamed: 0',inplace=True)
print(df.head())

print("\nData Cleaning")

print(df.isnull())
print("\n")
print(df.isnull().sum())
sns.heatmap(df.isnull(),yticklabels=False)
plt.show()

print("\nSeperating columns for classification")

X=df[['title','text']]  # Independent Variables
y=df[['label']]         # Dependent Variable

print("Train/Test Split")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=4)
print('Train set:\nPredictor:',X_train.shape,'\nPredicted:',y_train.shape)
print('\nTest set:\nPredictor:',X_test.shape,'\nPredicted:',y_test.shape)

print("\nModel Fitting")

print("\nCreating vector 1 for \'title\' column:")

tf1=TfidfVectorizer(stop_words='english',max_df=0.7)
X_trainTF1=tf1.fit_transform(X_train['title'])
X_testTF1=tf1.transform(X_test['title'])
pac1=PassiveAggressiveClassifier()
pac1.fit(X_trainTF1,y_train)
pred1=pac1.predict(X_testTF1)
print('\nPredicted values of column \'title\':\n')
print(pred1)

print("\nCreating vector 2 for \'text\' column:")

tf2=TfidfVectorizer(stop_words='english',max_df=0.7)
X_trainTF2=tf2.fit_transform(X_train['text'])
X_testTF2=tf2.transform(X_test['text'])
pac2=PassiveAggressiveClassifier()
pac2.fit(X_trainTF2,y_train)
pred2=pac2.predict(X_testTF2)
print('\nPredicted values of column \'text\':\n')
print(pred2)

print("\nEvaluation")

print("\nAccuracy of \'title\' column")

print('\nFor \'title\' column: ')
print('\nAccuracy: ',metrics.accuracy_score(y_test,pred1))
print('Classification Report:\n',classification_report(y_test,pred1))
mat1=confusion_matrix(y_test,pred1)
print('\nConfusion Matrix:\n')
print(mat1)

print("\nAccuracy of \'text\' column")

print('\nFor \'text\' column: ')
print('\nAccuracy: ',metrics.accuracy_score(y_test,pred2))
print('Classification Report:\n',classification_report(y_test,pred2))
mat2=confusion_matrix(y_test,pred2)
print('\nConfusion Matrix:\n')
print(mat2)

print("\nVisualizing confusion matrices")
print("You will need to close the present visualization to continue")

print("Confusion matrix 1:")
print("Confusion Matrix for \'title\' column!")
sns.heatmap(mat1, annot=True,cmap='Blues')
plt.show()

print("\nConfusion Matrix 2")
print("Confusion Matrix for \'text\' column!")
sns.heatmap(mat2, annot=True,cmap='Blues')
plt.show()

print("\nObservation:")

print("We see that the 'text' column, after appliying classification, gives a way better accuracy than the 'title' column.")

print("\nResult")

print("The 'text' column qualifies to be the better choice so as to identify REAL or FAKE news.")

