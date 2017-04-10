import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def svm (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30 , random_state = 0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = SVC(kernel = 'linear', random_state = 0)
    # Kernel cam be linear, poly, rbf, sigmoid
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    return X_train,X_test,y_train,y_test,y_pred,classifier,cm


   
