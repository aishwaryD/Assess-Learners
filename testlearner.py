import math
import sys

import numpy as np

import RTLearner as rtl
import DTLearner as dt
import BagLearner as bl

import LinRegLearner as lrl
import InsaneLearner as il

import time
import matplotlib.pyplot as plt


# Experiments

def exp1_ques1_2(Xtrain, Ytrain, Xtest, Ytest):

    rsmes_in_samples = []
    rsmes_out_samples = []

    for leaf_size in range(1, 101):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)

        Ypred_in_sample = learner.query(Xtrain)
        rmse_in = math.sqrt(((Ytrain - Ypred_in_sample) ** 2).sum() / Ytrain.shape[0])

        Ypred_out_sample = learner.query(Xtest)
        rmse_out = math.sqrt(((Ytest - Ypred_out_sample) ** 2).sum() / Ytest.shape[0])

        rsmes_in_samples.append(rmse_in)
        rsmes_out_samples.append(rmse_out)

    xi = range(1, 101)
    plt.xticks(np.insert(np.arange(5, 101, step=5), 0, 1))
    plt.xlabel("Leaf size")
    plt.ylabel("Rmse")
    plt.title("Overfitting with respect to leaf_size | DT")
    plt.plot(xi, rsmes_in_samples, label="in sample")
    plt.plot(xi, rsmes_out_samples, label="out of sample")
    plt.grid()
    plt.legend()
    plt.savefig("exp1_ques1.png")
    plt.clf()

    xi = range(1, 21)
    plt.xticks(np.insert(np.arange(5, 21, step=5), 0, 1))
    plt.xlabel("Leaf size")
    plt.ylabel("Rmse")
    plt.title("Direction of Overfitting | DT")
    plt.plot(xi, rsmes_in_samples[:20], label="in sample")
    plt.plot(xi, rsmes_out_samples[:20], label="out of sample")
    plt.grid()
    plt.legend()
    plt.savefig("exp1_ques2.png")
    plt.clf()


def exp2_ques1_2(Xtrain, Ytrain, Xtest, Ytest):

    rsmes_in_samples = []
    rsmes_out_samples = []

    for leaf_size in range(1, 101):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=30, boost=False,
                                verbose=False)
        learner.add_evidence(Xtrain, Ytrain)

        Ypred_in_sample = learner.query(Xtrain)
        rmse_in = math.sqrt(((Ytrain - Ypred_in_sample) ** 2).sum() / Ytrain.shape[0])

        Ypred_out_sample = learner.query(Xtest)
        rmse_out = math.sqrt(((Ytest - Ypred_out_sample) ** 2).sum() / Ytest.shape[0])

        rsmes_in_samples.append(rmse_in)
        rsmes_out_samples.append(rmse_out)

    xi = range(1, 101)
    plt.xticks(np.insert(np.arange(5, 101, step=5), 0, 1))
    plt.xlabel("Leaf size")
    plt.ylabel("Rmse")
    plt.title("Leaf Size and Overfitting in BagLearner with DT | 30 bags")
    plt.plot(xi, rsmes_in_samples, label="in sample")
    plt.plot(xi, rsmes_out_samples, label="out of sample")
    plt.grid()
    plt.legend()
    plt.savefig("exp2_ques1_2.png")
    plt.clf()


def exp3_ques1(Xtrain, Ytrain):

    dt_time = []
    rt_time = []

    for training in range(200, Xtrain.shape[0] + 1, 200):

        learner = dt.DTLearner(leaf_size=1, verbose=False)
        start_time = time.time()
        learner.add_evidence(Xtrain[:training], Ytrain[:training])
        end_time = time.time()
        execution_time = end_time - start_time
        dt_time.append(execution_time)

        learner = rtl.RTLearner(leaf_size=1, verbose=False)
        start_time = time.time()
        learner.add_evidence(Xtrain[:training], Ytrain[:training])
        end_time = time.time()
        execution_time = end_time - start_time
        rt_time.append(execution_time)

    xi = range(200, Xtrain.shape[0] + 1, 200)
    plt.xticks(np.arange(200, Xtrain.shape[0] + 1, step=400))
    plt.xlabel("Training length")
    plt.ylabel("Training time")
    plt.title("Decision Tree vs Random Tree")
    plt.plot(xi, dt_time, label="Decision Tree")
    plt.plot(xi, rt_time, label="Random Tree")
    plt.grid()
    plt.legend()
    plt.savefig("exp3_ques1.png")
    plt.clf()


def exp3_ques2(Xtrain, Ytrain, Xtest, Ytest):

    dt_out_sample_mae = []
    rt_out_sample_mae = []

    for leaf_size in range(1, 31):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)
        Ypred_out_sample = learner.query(Xtest)
        out_sample_mae = np.mean(np.abs((np.array(Ytest) - np.array(Ypred_out_sample))))
        dt_out_sample_mae.append(out_sample_mae * 100)

        learner = rtl.RTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)
        Ypred_out_sample = learner.query(Xtest)
        out_sample_mae = np.mean(np.abs((np.asarray(Ytest) - np.asarray(Ypred_out_sample))))
        rt_out_sample_mae.append(out_sample_mae * 100)

    xi = range(1, 31)
    plt.xticks(np.arange(1, 31, step=5))
    plt.xlabel("Leaf size")
    plt.ylabel("Mean Absolute Error")
    plt.plot(xi, dt_out_sample_mae, label="Decision Tree")
    plt.plot(xi, rt_out_sample_mae, label="Random Tree")
    plt.title("Decision Tree vs Random Tree on MAE")
    plt.grid()
    plt.legend()
    plt.savefig("exp3_ques2.png")
    plt.clf()


def exp3_ques3(Xtrain, Ytrain, Xtest, Ytest):

    dt_out_sample_mae = []
    bag_out_sample_mae = []

    for leaf_size in range(1, 31):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=7, boost=False,
                                verbose=False)
        learner.add_evidence(Xtrain, Ytrain)
        Ypred_out_sample = learner.query(Xtest)
        out_sample_mae = np.mean(np.abs((np.array(Ytest) - np.array(Ypred_out_sample))))
        bag_out_sample_mae.append(out_sample_mae * 100)

        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)
        Ypred_out_sample = learner.query(Xtest)
        out_sample_mae = np.mean(np.abs((np.array(Ytest) - np.array(Ypred_out_sample))))
        dt_out_sample_mae.append(out_sample_mae * 100)

    xi = range(1, 31)
    plt.xticks(np.arange(1, 31, step=5))
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error")
    plt.title("Decision Tree V/S Bagged Random Tree on MAE")
    plt.plot(xi, dt_out_sample_mae, label="Decision Tree")
    plt.plot(xi, bag_out_sample_mae, label="Bagged Random Tree")
    plt.grid()
    plt.legend()
    plt.savefig("exp3_ques3.png")
    plt.clf()


def exp3_ques4(Xtrain, Ytrain):

    dt_execution_time = []
    bag_execution_time = []

    for training in range(200, Xtrain.shape[0] + 1, 200):

        learner = dt.DTLearner(leaf_size=1, verbose=False)
        start_time = time.time()
        learner.add_evidence(Xtrain[:training], Ytrain[:training])
        end_time = time.time()
        execution_time = end_time - start_time
        dt_execution_time.append(execution_time)

        learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 1}, bags=7, boost=False, verbose=False)
        start_time = time.time()
        learner.add_evidence(Xtrain[:training], Ytrain[:training])
        end_time = time.time()
        execution_time = end_time - start_time
        bag_execution_time.append(execution_time)

    xi = range(200, Xtrain.shape[0] + 1, 200)
    plt.xticks(np.arange(200, Xtrain.shape[0] + 1, step=400))
    plt.xlabel("Training Length")
    plt.ylabel("Training Time")
    plt.title("Decision Tree vs Bagged Random Tree")
    plt.plot(xi, dt_execution_time, label="Decision Tree")
    plt.plot(xi, bag_execution_time, label="Bagged Random Tree")
    plt.grid()
    plt.legend()
    plt.savefig("exp3_ques4.png")
    plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])

    np.random.seed(903862212)

    data = np.array([list(map(str, s.strip().split(','))) for s in inf.readlines()])

    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]

    data = data.astype('float')

    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print(f"{testX.shape}")
    print(f"{testY.shape}")

    learner = dt.DTLearner(verbose=True)
    learner.add_evidence(trainX, trainY)  # train it
    print(learner.author())

    # evaluate in sample
    predY = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=trainY)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    predY = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=testY)
    print(f"corr: {c[0, 1]}")

    exp1_ques1_2(trainX, trainY, testX, testY)
    exp2_ques1_2(trainX, trainY, testX, testY)
    exp3_ques2(trainX, trainY, testX, testY)
    exp3_ques3(trainX, trainY, testX, testY)

    exp3_ques1(trainX, trainY)
    exp3_ques4(trainX, trainY)
