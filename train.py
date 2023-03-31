
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier




df = pd.read_csv("mobile_data.csv")

#set random seed
seed = 42

#Standardize features and split the data into train test
X = df.drop(columns=['price_range'],axis=1)
y = df['price_range']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test =  scalar.transform(X_test)

#Building Model 
clf =  LogisticRegression()
clf.fit(X_train,y_train)



#print accuracy to metrics.txt

acc = clf.score(X_test, y_test)

macro_averaged_f1score = f1_score(y_test,clf.predict(X_test), average = 'macro')
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc , "f1_score_macro" :macro_averaged_f1score}, outfile)
#plot a confusion matrix

disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("confusion_matrix.png")
