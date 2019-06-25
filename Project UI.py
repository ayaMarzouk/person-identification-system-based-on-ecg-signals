import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from os import system
import DSP_Project
from sklearn.metrics import accuracy_score


TestPath = None
def SignalBrowsing():
    global TestPath
    TestPath = askopenfilename(initialdir = r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\test',title = "Select ECG Signal",filetypes = [('Text files','*.txt')])
    TestPath = TestPath.replace('/','\\')


ECG_Signal, X_train, y_train = [], [], []
def LoadSignal():
    global ECG_Signal, TestPath, X_train, y_train
    with open(TestPath) as fileHandle:
        for line in fileHandle:
            line = line.rstrip()
            ECG_Signal.append(float(line))

    x = list(range(len(ECG_Signal)))
    plt.figure()
    plt.plot(x, ECG_Signal)
    plt.title('Input ECG Signal')
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    plt.show()



SubSegments = []
def PreProcessing():
    global ECG_Signal, SubSegments
    SubSegments = DSP_Project.PreProcessing(ECG_Signal)
    plt.show()


X_test, y_test = [], []
def FeatureExtraction():
    global SubSegments, X_test, y_test
    featureVector = DSP_Project.FeatureExtraction(SubSegments)
    print (len(featureVector))
    X_test, y_test = np.array(featureVector), np.array(y_test)
    X_test = X_test.reshape(X_test.shape[:2])
    _ = system('cls')
    plt.show()


def RunSVM():
    #_ = system('cls')
    global X_test, y_test, X_train, y_train, var1, var2
    kernel = None
    if var1.get():
        kernel = 'rbf'
    elif var2.get():
        kernel = 'linear'
    Classifier = DSP_Project.RunClassifier(X_train, y_train, 'SVM', kernel, None)
    Labely = int (TestPath.split('.')[0][-1]) - 1
    N = X_test.shape[0]
    y_test = np.array([Labely] * N)

    predictions = Classifier.predict(X_test)
    acc = round(accuracy_score(y_test,predictions),4)*100
    print ('Accuracy for Person using SVM with kernel {} is: {}%'.format(kernel,acc))


def RunKNN():
    #_ = system('cls')
    global  X_test, y_test, X_train, y_train, v
    Classifier = DSP_Project.RunClassifier(X_train, y_train, 'KNN', None, int(v.get()))
    Labely = int (TestPath.split('.')[0][-1]) - 1
    N = X_test.shape[0]
    y_test = np.array([Labely] * N)

    predictions = Classifier.predict(X_test)
    acc = round(accuracy_score(y_test,predictions),4)*100
    print ('Accuracy for Person using KNN using {} neighbors is: {}%'.format(int(v.get()),acc))


def RunGaussianNB():
    #_ = system('cls')
    global  X_test, y_test, X_train, y_train
    Classifier = DSP_Project.RunClassifier(X_train, y_train, 'GaussianNB', None, None)
    Labely = int (TestPath.split('.')[0][-1]) - 1
    N = X_test.shape[0]
    y_test = np.array([Labely] * N)

    predictions = Classifier.predict(X_test)
    acc = round(accuracy_score(y_test,predictions),4)*100
    print ('Accuracy for Person using GaussianNB is: {}%'.format(acc))


def RunRandomForest():
    #_ = system('cls')
    global  X_test, y_test, X_train, y_train
    Classifier = DSP_Project.RunClassifier(X_train, y_train, 'RandomForest', None, None)
    Labely = int (TestPath.split('.')[0][-1]) - 1
    N = X_test.shape[0]
    y_test = np.array([Labely] * N)

    predictions = Classifier.predict(X_test)
    acc = round(accuracy_score(y_test,predictions),4)*100
    print ('Accuracy for Person using RandomForest is: {}%'.format(acc))


def from_rgb_to_hexa(rgb):
    return "#%02x%02x%02x" % rgb


root = Tk()
root.title('ECG Based Biometrics')
root.geometry('962x602')

image = PhotoImage(file='img.png')
im = Label(root, image=image)
im.pack()


BrowseingB = Button(root, text='Browse ECG Signal', font=('Tempus Sans ITC',20), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=SignalBrowsing)
BrowseingB.place(relx=0.779, rely=0.01, width=210 ,height=60)

LoadingB = Button(root, text='Load ECG Signal', font=('Tempus Sans ITC',18), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=LoadSignal)
LoadingB.place(relx=0.779, rely=0.11, width=210 ,height=60)

PreProcessingB = Button(root, text='PreProcessing', font=('Tempus Sans ITC',22), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=PreProcessing)
PreProcessingB.place(relx=0.779, rely=0.21, width=210 ,height=60)

FeatureExtractionB = Button(root, text='FeatureExtraction', font=('Tempus Sans ITC',20), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=FeatureExtraction)
FeatureExtractionB.place(relx=0.779, rely=0.31, width=210 ,height=60)

var1 = IntVar()
LinearChk = Checkbutton(root, text='RBF', variable=var1)
LinearChk.place(relx=0.8, rely=0.515, height=30)
var2 = IntVar()
RBFChk = Checkbutton(root, text='Linear', variable=var2)
RBFChk.place(relx=0.9, rely=0.515, height=30)

SVMB = Button(root, text='Predict using SVM', font=('Tempus Sans ITC',20), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=RunSVM)
SVMB.place(relx=0.779, rely=0.41, width=210 ,height=60)


N_neighbors = Label(root, text='n_neighbors')
N_neighbors.place(relx=0.8, rely=0.675, height=30)
v = StringVar()
e1 = Entry(root, textvariable=v)
e1.place(relx=0.9, rely=0.675, width=75, height=30)
KNNB = Button(root, text='Predict using KNN', font=('Tempus Sans ITC',20), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=RunKNN)
KNNB.place(relx=0.779, rely=0.57, width=210 ,height=60)



GaussianB = Button(root, text='Predict using Gaussian', font=('Tempus Sans ITC',16), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=RunGaussianNB)
GaussianB.place(relx=0.779, rely=0.75, width=210 ,height=60)


RandomFB = Button(root, text='Predict using RF', font=('Tempus Sans ITC',20), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=RunRandomForest)
RandomFB.place(relx=0.779, rely=0.85, width=210 ,height=60)

tk.mainloop()