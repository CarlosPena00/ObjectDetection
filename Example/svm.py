import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC

def svm (X,y,Kernel = 'rbf'):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30 , random_state = 0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = SVC(kernel = Kernel)  # Kernel cam be linear, poly, rbf, sigmoid
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print cm
    print (cm[0,0]+cm[1,1])/float(cm.sum())
    return X_train,X_test,y_train,y_test,y_pred,classifier,cm,sc_X

def svmLinear (X,y,Ce=0.01):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30 , random_state = 0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = LinearSVC(C=Ce)
    # Kernel cam be linear, poly, rbf, sigmoid
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print cm
    print (cm[0,0]+cm[1,1])/float(cm.sum())
    return X_train,X_test,y_train,y_test,y_pred,classifier,cm,sc_X

def randomF(X,y, N = 100,theads = 3):
	print "----- Start To Split -----"
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30,  random_state = 0)
	sc_X = StandardScaler()
	print "----- Start Classifier -----"
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	classifier = RandomForestClassifier(n_estimators=N,n_jobs=theads,criterion='entropy')
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test,y_pred)
	print cm
	print (cm[0,0]+cm[1,1])/float(cm.sum())

	return X_train,X_test,y_train,y_test,y_pred,classifier,cm,sc_X


"""   
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.imsave("Save.jpg")
"""
    

   
