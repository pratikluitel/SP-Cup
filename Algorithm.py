import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
import json
import math
import pickle

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = np.zeros([10,1])
        self.new_m = np.zeros([10,1])
        self.old_s = np.zeros([10,1])
        self.new_s = np.zeros([10,1])

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros([10,1])
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros([10,1])

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else np.zeros([10,1])

    def standard_deviation(self):
        return math.sqrt(self.variance())

def normalize_test_data(Data, mean, std, params = ['x','y','z','x.1','y.1','z.1','x.2','y.2','z.2']):
    imuParamsData = Data.loc[:, params]
    normImuData = (imuParamsData - mean)/std
    return normImuData

def find_theta_score(Data,dims=1):
    """
    Converts n dimensions to a lower dimensional score,
    for easier visualization
    """
    pca = PCA(n_components=dims)
    pca.fit(Data)
    theta_score = pca.transform(Data)
    return theta_score


###Code remaining: take one array with 9 params as input
def train(Data):
    train.arr.append(Data)

train.arr = []

def stopTrain(filename):
    """
    Stops training and saves the model as filename.sav
    also saves the threshold, mean and standard deviation
    in a json file of the same name.
    """
    z = find_theta_score(train.arr)
    lof = KNN()
    lof.fit(z)
    scores = lof.decision_scores_
    
    # save model
    with open(filename+'.sav', 'wb') as savf:
        pickle.dump(lof, savf)
    
    mean = scores.mean()
    stdev = scores.std()
    thres = mean+18*stdev
    params = {}
    params['mean'] = mean
    params['std'] = stdev
    params['threshold'] = thres
    with open(filename+'.json', 'wb') as jsonf:
        json.dump(params,jsonf)



def start_detection(filename):
    """
    """
    global detectorlof 
    detectorlof = pickle.load(open(filename, 'rb'))



def detect_abnormalities(Data):
    """
    Data.header.stamp = timestamp in ns
    Data.linear_acceleration has x, y, z, w values as .x,.y,.z
    Data.angular_velocity has x, y and z values as .x,.y,.z
    Data.orientation has x, y and z values as .x,.y,.z,.w
    """
    score = detectorlof.decision_function(find_theta_score(Data,2))
    return score

