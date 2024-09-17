import numpy as np
import pandas as pd
import os 

df = pd.read_csv('/Users/hunjunsin/Desktop/python/Kaggle/adclickpredict/ad_click_dataset.csv')
df.head()

df.info()
df.isnull().sum()
df.describe()
df['gender'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'Random Forest': RandomForestClassifier(random_state=10),
    'Gradient Boosting': GradientBoostingClassifier(random_state=10),
    'Ada Boost': AdaBoostClassifier(random_state=10),
    'Extra Trees': ExtraTreesClassifier(random_state=10),
    'XGBoost': XGBClassifier(random_state=10),
    'Cat Boost': CatBoostClassifier(random_state=10, verbose=False),
    'Light GBM': LGBMClassifier(random_state=10, verbose=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=10),
    'Logistic Regression': LogisticRegression(random_state=10)
}

df = df.drop(['full_name'], axis = 1)
df.head()
df.isnull().sum()
df['age'] = df['age'].fillna(df['age'].mean())
df['gender'] = df['gender'].fillna('Unknown')
df['device_type'] = df['device_type'].fillna('Unknown')
df['ad_position'] = df['ad_position'].fillna('Unknown')
df['browsing_history'] =df['browsing_history'].fillna('Unknown')
df['time_of_day'] =df['time_of_day'].fillna('Unknown')

df.isnull().sum()
df['time_of_day'].head()
le = LabelEncoder()
df['gender'] =le.fit_transform(df['gender'])
df['device_type'] = le.fit_transform(df['device_type'])
df['ad_position'] = le.fit_transform(df['ad_position'])
df['browsing_history'] = le.fit_transform(df['browsing_history'])
df['time_of_day'] = le.fit_transform(df['time_of_day'])

df.info()
df['time_of_day'].head()

X = df.drop('click', axis =1)
y = df['click']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

for name, model in models.items():
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-"* 20)
    
from sklearn.model_selection import GridSearchCV

param_grids = {
    'Decision Tree': {
       'max_depth': [3, 5, 10, 15],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 5, 10]
    },
    'XGBoost': {
       'max_depth': [3, 5, 10],
        'learning_rate': [0.1, 0.5, 1],
        'n_estimators': [50, 100, 200]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
       'max_depth': [3, 5, 10],
       'min_samples_split': [2, 5, 10]
    }
}

for name, model in models.items():
    if name not in param_grids:
        continue
    
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', verbose =1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    
    print(f"Best Parameters for : {name}: {grid_search.best_params_}")
    print(f"Best score for : {name}: {grid_search.best_score_}")
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification_Report: ")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("*"*30)