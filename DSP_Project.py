import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy.fftpack import dct
from joblib import dump, load
import cv2
import os
from math import *
from cmath import *
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from os import system
_ = system('cls')


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(EcgSignal, lowcut = 1, highcut = 40, fs=1000 , order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order= order)
    EcgFilterdSignal = lfilter(b, a, EcgSignal)
    return EcgFilterdSignal


def DivideSegments(EcgSignal):
    SubSegments = []
    N = len(EcgSignal)
    i = 0
    while(i+820)<N:
        SubSegments.append(EcgSignal[i:i+820])
        i+=820

    return SubSegments


def PreProcessing(EcgSignal):
    x = list(range(len(EcgSignal)))

    EcgSignal = np.array(EcgSignal)
    MeanValue = np.mean(EcgSignal)
    MeanRemoved = EcgSignal - MeanValue
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(x, MeanRemoved)
    plt.title('After Applying Mean Removal on Input ECG Signal')
    plt.xticks([])


    Filterd = butter_bandpass_filter (MeanRemoved)
    plt.subplot(4, 1, 2)
    plt.plot(x, Filterd)
    plt.title('Filtered ECG Signal')
    plt.xticks([])


    Normalized = cv2.normalize(Filterd, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
    plt.subplot(4, 1, 3)
    plt.plot(x, Normalized)
    plt.title('Normalized ECG Signal')

    SubSegments = DivideSegments(Normalized)
    plt.subplot(4, 1, 4)
    plt.plot(x[:820], SubSegments[0])
    plt.title('First four heartbeats in ECG Signal')
    return SubSegments


def DCT(Segment):
    outSignal = []
    N = Segment.size
    for k in range(N):
        s = 0
        for n in range(N):
            s+= (Segment[n]*cos((pi/N)*(n + 0.5)*k).real)
        outSignal.append(s)

    outSignal = np.array(outSignal)
    NonZeroCoeff = [element!=0 for element in outSignal]
    outSignal = outSignal[NonZeroCoeff]
    return outSignal


def AutoCorrelation (Segment):
    N = Segment.size
    tcpy = Segment.copy()
    ComputedNormalized = []
    for i in range(N):
        r11 = np.sum(np.multiply(tcpy,Segment))
        term1 = np.sum(np.power(tcpy,2.0))
        term2 = np.sum(np.power(Segment,2.0))
        p11 = r11/(np.sqrt(term1*term2))
        ComputedNormalized.append(p11)
        firstOne = Segment[0]
        Segment = np.array(list(Segment[1:]) + [firstOne])
    
    return np.array(ComputedNormalized).real


def FeatureExtraction(SubSegments):
    featureVector = []
    for i, Segment in enumerate(SubSegments):
        CorrelatedSignal = AutoCorrelation (Segment)
        DCT_Signal = DCT(CorrelatedSignal)
        if not i:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(range(CorrelatedSignal.size), CorrelatedSignal)
            plt.title('Autocorrelation for the first Segment')
            
            plt.subplot(2, 1, 2)
            plt.plot(range(DCT_Signal.size), DCT_Signal)
            plt.title('DCT_Signal for the first Autocorrelated Segment')
        featureVector.append(DCT_Signal)

    return featureVector


def RunClassifier(x_Train, y_train, classifier, kernel, neighbors):
    Classifer = None
    if classifier=='SVM':
        Classifer = svm.SVC(kernel=kernel)
    elif classifier=='KNN':
        Classifer = KNeighborsClassifier(n_neighbors=neighbors)
    elif classifier=='GaussianNB':
        Classifer = GaussianNB()
    elif classifier=='BernoulliNB':
        Classifer = BernoulliNB()
    elif classifier=='RandomForest':
        Classifer = RandomForestClassifier(n_jobs=-1)

    Classifer.fit (x_Train, y_train)
    return Classifer


def Test(TestDirs,Classifer):
    FullX_test, Fully_test = [], []
    if os.path.exists('FullX_test.npy') and os.path.exists('Fully_test.npy'):
        FullX_test = np.load('FullX_test.npy')
        Fully_test = np.load('Fully_test.npy')


    else:
        for test_dir in TestDirs:
            SubSegments = PreProcessing(test_dir)
            featureVector = FeatureExtraction(SubSegments)
            Label = int (test_dir.split('.')[0][-1]) - 1
            N = len (featureVector)
            y_test = [Label]* N

            X_test, y_test = np.array(featureVector), np.array(y_test)
            X_test = X_test.reshape(X_test.shape[:2])
            FullX_test.append(X_test)
            Fully_test.append(y_test)

        np.save('FullX_test.npy',FullX_test)
        np.save('Fully_test.npy',Fully_test)

    ik = 1
    for X_test, y_test in zip(FullX_test,Fully_test):
        predictions = Classifer.predict(X_test)
        acc = round(accuracy_score(y_test,predictions),4)*100
        print ('Accuracy for Person # {} is: {}%'.format(ik,acc))
        ik+=1




TrainDirs  = [r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\train\s1.txt',
              r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\train\s2.txt',
              r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\train\s3.txt']

TestDirs = [r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\test\s1.txt',
              r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\test\s2.txt',
              r'E:\Scientific\sem 2\Signal\Project\Data\biometrics\test\s3.txt']

#Classifer = Run(TrainDirs)
#Test(TestDirs,Classifer)
