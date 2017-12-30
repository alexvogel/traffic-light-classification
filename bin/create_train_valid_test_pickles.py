import sys
import os
import argparse
import pickle
import cv2
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--toc_file", metavar='PATH', action='store', required=True, help="toc file")
    parser.add_argument("--outdir", metavar='PATH', type=str, required=True, default="", help="output directory for pickle files of train, valid and test image plus label files")
    args = parser.parse_args()

    # create directories
    if os.path.exists(args.outdir):
        print("error: delete or rename " + args.outdir)
        sys.exit(1)
    else:
        os.mkdir(args.outdir)

    # check if toc file exists
    if not os.path.isfile(args.toc_file):
        print("error: file does not exist: " + args.toc_file)
        sys.exit(1)

    # read toc file
    with open(args.toc_file, "r") as f:
        content = f.readlines()

    # remove first line
    content = content[1:]

    # remove newlines
    content = [x.strip() for x in content]

    # create dict path_label
    path_label = {}

    # data store
    #features = np.ndarray(shape=( 1, 32, 32, 3))
    features_list = []
    labels_list = []

    for line in content:

        label, path = line.split(",")
        path_label[path] = label

        # create abs path from joining the toc path and the rel image path
        abs_path_to_image = os.path.abspath(os.path.join(os.path.dirname(args.toc_file), path))
        
        # read in as numpy array
        img = cv2.imread(abs_path_to_image)

        # append to lists
        features_list.append(img)
        labels_list.append(label)

    # log shapes
    features = np.asarray(features_list)
    print("features shape: " + str(features.shape))
    labels = np.asarray(labels_list)
    print("labels shape: " + str(labels.shape))

    # put away 10% for testing, 20% for validating and 70% for training
    features_train = features[int(len(features)*0.3):]
    features_test  = features[:int(len(features)*0.1)]
    features_valid = features[int(len(features)*0.1):int(len(features)*0.3)]
    
    labels_train = labels[int(len(labels)*0.3):]
    labels_test  = labels[:int(len(labels)*0.1)]
    labels_valid = labels[int(len(labels)*0.1):int(len(labels)*0.3)]
    
    print("features_train shape: " + str(features_train.shape))
    print("features_valid shape: " + str(features_valid.shape))
    print("features_test shape: " + str(features_test.shape))

    print("labels_train shape: " + str(labels_train.shape))
    print("labels_valid shape: " + str(labels_valid.shape))
    print("labels_test shape: " + str(labels_test.shape))

    # create dict 
    dict_train = {}
    dict_train['features'] = features_train
    dict_train['labels'] = labels_train

    dict_valid = {}
    dict_valid['features'] = features_valid
    dict_valid['labels'] = labels_valid

    dict_test = {}
    dict_test['features'] = features_test
    dict_test['labels'] = labels_test

    # and write as pickle
    print("writing pickle file for training " + args.outdir + "/train.p")
    pickle.dump(dict_train, open(args.outdir + "/train.p", "w"))
    
    print("writing pickle file for validating " + args.outdir + "/valid.p")
    pickle.dump(dict_valid, open(args.outdir + "/valid.p", "w"))
    
    print("writing pickle file for testing " + args.outdir + "/test.p")
    pickle.dump(dict_test, open(args.outdir + "/test.p", "w"))

if __name__ == '__main__':
    main()
