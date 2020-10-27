# Hilarity classifier Neural Network
# This program performs binary classification of English one-liners as funny (output 1) or not funny (output 0).

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import random, math
import sys
import torch
from torch.autograd import Variable
import torch.optim as optim

from model.nn import HilarityNN
import utils

random.seed(0)

def parse_data_set(fname):
    lines = utils.read_file(fname)
    dataset = []
    for idx, line in enumerate(lines):
        one_liner, label = line.split("-*-*-")
        point = {}
        point["one_liner"] = textprocessing.cleanTextString(one_liner.strip())
        point["label"] = label.strip()
        dataset.append(point)
    return dataset

def get_vocab(dataset):
    words = set()
    for point in dataset:
        for word in point["one_liner"].split():
            words.add(word)
    return words

def featurize(dataset):
    output_map = {"meme" : [1, 0], "headline" : [0, 1]}
    features = []
    outputs = []
    for point in dataset:
        if len(point["one_liner"]) == 0:
            continue
        features.append(utils.get_awe(point["one_liner"].split()))
        outputs.append(output_map[point["label"]])
    return features, outputs

def validation(X, Y, model, opt, criterion):
    total_trues = 0
    total_positives = 0
    total_true_positives = 0
    total = correct = 0
    with torch.no_grad():
        for idx in range(0, Y.size(0)):
            x_val = X[idx]
            y_val = Y[idx]
            y_hat = model(x_val)
            _, y_val = y_val.max(0)
            _, y_hat = y_hat.max(0)
            if y_hat == y_val:
                correct += 1
            total += 1
    print(correct, total)

def train_epoch(X, Y, model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for idx in range(0, Y.size(0), batch_size):
        x_batch = Variable(X[idx : idx + batch_size, :])
        y_batch = Variable(Y[idx : idx + batch_size, :])
        opt.zero_grad()
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        opt.step() 
        losses.append(loss.data.numpy())
    return [sum(losses)/float(len(losses))]

def train(X_train, Y_train, X_test, Y_test):
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)
    net = HilarityNN(X_train.shape[1], Y_train.shape[1], num_neurons_per_layer=args.nn_neurons)
    opt = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    train_f1 = []
    val_f1 = []
    e_losses = []
    for idx in range(args.num_epochs):
        e_losses += train_epoch(X_train, Y_train, net, opt, criterion, args.batch_size)
    validation(X_test, Y_test, net, opt, criterion)
    plt.plot(e_losses)
    plt.show()
    torch.save(net.state_dict(), os.getcwd() + "/net")
    torch.save(opt.state_dict(), os.getcwd() + "/opt")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--w2v_path", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=float, default=250)
    parser.add_argument("--batch_size", type=float, default=32)
    parser.add_argument("--nn_neurons", type=int, default=1000)
    args = parser.parse_args()

    dataset = parse_data_set(args.data)
    utils.save_w2v(args.w2v_path, get_vocab(dataset))
    features, outputs = featurize(dataset)
    train_inputs, train_outputs, val_inputs, val_outputs = utils.cv_split(features, outputs, 0.8)
    train(args, train_inputs, train_outputs, val_inputs, val_outputs)
	
