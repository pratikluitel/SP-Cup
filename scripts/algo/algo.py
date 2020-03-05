import numpy as np
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
import json
import pickle

from flask_socketio import SocketIO, emit
import time
import random
from PIL import Image
import numpy as np

socketio = SocketIO(message_queue='redis://')

def find_theta_score(Data,pca,dims=1):
    """
    Converts n dimensions to a lower dimensional score,
    for easier visualization
    """
    theta_score = pca.transform(Data)
    return theta_score


def start_train():
    pass

def train(Data):

    in_data = [
        Data.orientation.x, Data.orientation.y, Data.orientation.z,
        Data.orientation.w, Data.angular_velocity.x, Data.angular_velocity.y,
        Data.angular_velocity.z, Data.linear_acceleration.x,
        Data.linear_acceleration.y, Data.linear_acceleration.z
    ]

    input_data = np.array(in_data)

    train.arr.append(input_data)
    print(len(train.arr))


def stop_train(filename):
    """
    Stops training and saves the model as filename.sav
    also saves the threshold, mean and standard deviation
    in a json file of the same name. Also saves the pca model
    """
    pca = PCA(n_components=3)
    pca.fit(np.array(train.arr))
    with open(filename+'pca.sav', 'wb') as savpca:
        pickle.dump(pca, savpca)
    z = find_theta_score(np.array(train.arr),pca)

    lof = KNN(n_neighbors=1)
    lof.fit(z)
    scores = lof.decision_scores_
    with open(filename+'knn.sav', 'wb') as savknn:
        pickle.dump(lof, savknn)

    mean = scores.mean()
    stdev = scores.std()
    thres = mean+18*stdev
    params = {}
    params['mean'] = mean
    params['std'] = stdev
    params['threshold'] = thres
    with open(filename+'.json', 'w') as jsonf:
        json.dump(params,jsonf)
    
    print()
    print("Training Completed")


def start_detection(filename):
    """
    """
    global detectorlof
    global params
    global pca
    pca = pickle.load(open(filename+'pca.sav', 'rb'))
    detectorlof = pickle.load(open(filename+'knn.sav', 'rb'))
    params = json.load(open(filename+'.json'))

def detect_abnormalities(Data):
    """
    Data.header.stamp = timestamp in ns
    Data.linear_acceleration has x, y, z, w values as .x,.y,.z
    Data.angular_velocity has x, y and z values as .x,.y,.z
    Data.orientation has x, y and z values as .x,.y,.z,.w
    """
    in_data = [[
        Data.orientation.x, Data.orientation.y, Data.orientation.z,
        Data.orientation.w, Data.angular_velocity.x, Data.angular_velocity.y,
        Data.angular_velocity.z, Data.linear_acceleration.x,
        Data.linear_acceleration.y, Data.linear_acceleration.z
    ]]

    input_data = np.array(in_data).reshape(1, -1)
    score = detectorlof.decision_function(find_theta_score(input_data,pca))[0]
    abnormal = False
    ano = 'Non-Anomalous'
    if score > params['threshold']:
        abnormal = True
        ano = 'Anomalous'

    theta = find_theta_score(in_data,pca,dims=3)
    imu_proc = find_theta_score(in_data,pca,dims=1)[0][0]

    data = {
        'label': ano,
        'timestamp': Data.header.stamp.to_sec(),
        'score': score,
        'imu0': theta[0][0],
        'imu1': theta[0][1],
        'imu2': theta[0][2],
        'imu_proc': imu_proc
    }
    socketio.emit('new data', data, namespace='/dashboard')

    time.sleep(0.2)

    return score, abnormal


def stop_detection():
    pass

train.arr = []