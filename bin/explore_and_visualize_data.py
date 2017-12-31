import sys
import os
import argparse
import pickle
import random
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def list_label_map(label_map_file, outdir):

    id_name = {}
    
    labelNames = []

    with open(label_map_file) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            id_name[int(row['class_id'])] = row['class_name']
            labelNames.append(row['class_name'])
    
    # write to file
    f = open(outdir + "/summary.txt", "a")

    for id in id_name.keys():
        string = str(id)+": "+id_name[id]
        print(string)
        f.write(string + "\n")

    f.close()

    return id_name

def load_data(training_file, validation_file, testing_file):

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def basic_summary(X_train, y_train, X_valid, y_valid, X_test, y_test, outdir):
    # TODO: Number of training examples
    #print(train)
    n_train = len(X_train)

    # Number of validation examples
    n_valid = len(X_valid)

    # TODO: Number of testing examples.
    n_test = len(X_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = np.array(X_train[1]).shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(set(y_test))

    # write to file
    f = open(outdir + "/summary.txt", "a")

    string = "Number of training examples = " + str(n_train)
    print(string)
    f.write(string + "\n")

    string = "Number of validation examples = " + str(n_valid)
    print(string)
    f.write(string + "\n")

    string = "Number of testing examples = " + str(n_test)
    print(string)
    f.write(string + "\n")

    string = "Image data shape = " + str(image_shape)
    print(string)
    f.write(string + "\n")

    string = "Number of classes = " + str(n_classes)
    print(string)
    f.write(string + "\n")

    f.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--label_map", metavar='PATH', action='store', required=True, help="label map file in csv format")
    parser.add_argument("--outdir", metavar='PATH', type=str, required=True, default="", help="output directory for exploratory files")
    parser.add_argument("--train_file", metavar='PATH', type=str, required=True, default="", help="pickled data for training")
    parser.add_argument("--valid_file", metavar='PATH', type=str, required=True, default="", help="pickled data for validating")
    parser.add_argument("--test_file", metavar='PATH', type=str, required=True, default="", help="pickled data for testing")

    args = parser.parse_args()

    # check if pickled files are present
    for file in (args.train_file, args.valid_file, args.test_file):
        if not os.path.isfile(file):
            print("error: file does not exist: " + file)
            sys.exit(1)

    # create outdir
    if os.path.exists(args.outdir):
        print("rename or delete " + args.outdir)
        sys.exit(1)
    else:
        os.mkdir(args.outdir)

    # parse and print label map
    id_name = list_label_map(args.label_map, args.outdir)

    # load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(args.train_file, args.valid_file, args.test_file)

    print("shape of training features: " + str(X_train.shape))
    print("shape of training labels: " + str(y_train.shape))

    print("shape of validation features: " + str(X_valid.shape))
    print("shape of validation labels: " + str(y_valid.shape))

    print("shape of testing features: " + str(X_test.shape))
    print("shape of testing labels: " + str(y_test.shape))

    basic_summary(X_train, y_train, X_valid, y_valid, X_test, y_test, args.outdir)

    # plotting 6 random traffic lights with #nr and #label
    for i in range(1, 7):
        nr = random.randrange(0, len(X_train)-1, 1)
        subplot = plt.subplot(2, 3, i) # equivalent to: plt.subplot(2, 2, 1)
        subplot.set_xticks(())
        subplot.set_yticks(())
    
        plt.imshow(X_train[nr], cmap="hot")


        #print("nr=" + str(nr) + " y_train[nr]=" + str(y_train[nr]))
        #print(str(id_name.keys()))
        #print("id_name[y_train[nr]]=" + id_name[y_train[nr]])

        plt.title('count='+str(nr) + ' label='+str(y_train[nr])+'\n'+id_name[y_train[nr]], fontsize=8)

    plt.savefig(args.outdir + '/samples.png')
    plt.show()
    
    n, bins, patches = plt.hist(y_train, len(id_name.keys()))
    plt.savefig(args.outdir + '/histogram_training_data.png')
    #plt.show()


if __name__ == '__main__':
    main()
