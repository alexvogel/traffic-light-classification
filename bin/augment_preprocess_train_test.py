import sys
import os
import argparse
import time
import datetime
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import csv


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

def get_label_map(label_map_file):

    id_name = {}
    
    labelNames = []

    with open(label_map_file) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            id_name[int(row['class_id'])] = row['class_name']
            labelNames.append(row['class_name'])
    
    for id in id_name.keys():
        string = str(id)+": "+id_name[id]
        print(string)


    return id_name

def createDictIndex(x, y):
    '''
    x: list of images
    y: list of label
    returns list|dict of sign names
    '''

    dictIndex = {}
    
    setIndex = set(y)
    
    for index in setIndex:
        dictIndex[index] = []
    
    for i in range(len(y)):
        dictIndex[y[i]].append(i)
    
    return dictIndex


def rotate_scale(x, y, angle, scale):
    
    '''
    data augmentation
    x: list of images
    y: list of label
    angle: rotation angle
    scale: scale factor
    returns x, y with added new (augmented) data
    '''
    
    print("scale:", scale)
    augmented_x = []
    augmented_y = []
    
    center = (len(x[0])/2, len(x[0][0])/2 );
    M = cv2.getRotationMatrix2D( center, angle, scale );
    
    dictIndex = createDictIndex(x, y)
    
    for indexSignType in dictIndex.keys():
        
        count = len(dictIndex[indexSignType])
#        if count > TARGET_COUNT_PER_SIGN_TYPE:
#            print("sign type", indexSignType, ": already more than", TARGET_COUNT_PER_SIGN_TYPE, " images (", count, ")")
        if count < TARGET_COUNT_PER_SIGN_TYPE:
#            print("sign type", indexSignType, ": count", count, "generating rotated images")
            for indexImage in dictIndex[indexSignType]:
                img = x[indexImage]
                rows, cols, depth = img.shape
                
                rot_dst = np.zeros( img.shape)
                
                dst = cv2.warpAffine( img, M, (cols, rows) );
                
                augmented_x.append(dst)
                augmented_y.append(indexSignType)
    
    if (len(x.shape) == len(np.array(augmented_x).shape) and len(y.shape) == len(np.array(augmented_y).shape)):
        x = np.vstack([x, np.array(augmented_x)])
        y = np.concatenate((y, np.array(augmented_y)), axis=0)
    else:
        print("no images added")
        
    return(x, y)

def translate(x, y, trans_h, trans_v):
    '''
    data augmentation
    x: list of images
    y: list of label
    trans_h: translation horizontal
    trans_v: translation vertical
    returns x, y with added new (augmented) data
    '''
    
    augmented_x = []
    augmented_y = []
    
    center = (len(x[0])/2, len(x[0][0])/2 );
    M = np.float32([[1,0,trans_h],[0,1,trans_v]])
    
    dictIndex = createDictIndex(x, y)
    
    for indexSignType in dictIndex.keys():
        
        count = len(dictIndex[indexSignType])
#        if count > TARGET_COUNT_PER_SIGN_TYPE:
#            print("sign type", indexSignType, ": already more than", TARGET_COUNT_PER_SIGN_TYPE, " images (", count, ")")
        if count < TARGET_COUNT_PER_SIGN_TYPE:
#            print("sign type", indexSignType, ": count", count, "generating rotated images")
            for indexImage in dictIndex[indexSignType]:
                img = x[indexImage]
                rows, cols, depth = img.shape
                
                rot_dst = np.zeros( img.shape)
                
                dst = cv2.warpAffine( img, M, (cols, rows) );
                
                augmented_x.append(dst)
                augmented_y.append(indexSignType)
                
    if (len(x.shape) == len(np.array(augmented_x).shape) and len(y.shape) == len(np.array(augmented_y).shape)):
        x = np.vstack([x, np.array(augmented_x)])
        y = np.concatenate((y, np.array(augmented_y)), axis=0)
    else:
        print("no images added")

    return(x, y)

def augment(x, y):
    print("count images: ", len(x))
    print(x.shape)

    # Rotation
    for angle in (-1, 1, -2, 2):
        x, y = rotate_scale(x, y, angle, 1.05)
        print("count images: ", len(x))
   
    # Zoom in
    for scale in (1.05, 1.10):
        x, y = rotate_scale(x, y, 0., scale)
        print("count images: ", len(x))

    # Translate
    for transl in ([1., 0.], [-1., 0.], [0., 1.], [0., -1.]):
        x, y = translate(x, y, transl[0], transl[1])
        print("count images: ", len(x))

    return(x, y)

def rgb2gray(x):
    return np.dot(x[...,:3], [0.299, 0.583, 0.114])

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    
    x_norm = []

    for img in x:
        x_norm.append(img/255)
    
    return x_norm

def grayscale(x):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    x_gray = []
    for img in x:
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x_gray.append(img2)
        
    return np.array(x_gray)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def reshape(x):
    '''
    preprocess

    '''
    img2 = []
    for img in x:
        img_reshaped = img.reshape((32, 32, 1))
        img2.append(img_reshaped)
    return np.array(img2)

def one_hot_encode(x, n_classes):
    """
    preprocess
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : n_classes: Number of classes
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    all = []
    
    for label in x:
#        print(label)
        
        # create vector with zeros
        lbl_vec = np.zeros(n_classes)
        
        # set the appropriate value to 1.
        lbl_vec[label] = 1.
        
#        print(lbl_vec)
        
        all.append(lbl_vec)
    
#    print(all)
    return np.array(all)

def preprocess(x, y, n_classes):
    '''
    : x: features vector
    : y: labels vector
    '''
    x = rgb2gray(x)

    x = normalize(x)
    
    x = reshape(x)

    # Transform all labels in one_hot representation
    y = one_hot_encode(y, n_classes)
    
    return(x.astype(np.float32), y.astype(np.float32))


def myflatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    shape = x_tensor.get_shape().as_list()
    x_reshaped = tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])
    
#    print(x_reshaped.get_shape().as_list())
    
    return x_reshaped

def LeNet(x, keep_prob, n_classes):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = mu, stddev = sigma), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1, name='conv1')

    # Pooling. Input = 32x32x32. Output = 16x16x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2: Convolutional. Input = 16x16x32 Output = 16x16x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2, name='conv2')

    # dropout
    conv2 = tf.nn.dropout(conv2, keep_prob)
    
    # Pooling. Input = 16x16x64. Output = 8x8x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 3: Convolutional. Input = 8x8x64 Output = 8x8x128.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma), name='conv3_W')
    conv3_b = tf.Variable(tf.zeros(128))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    
    # Activation.
    conv3 = tf.nn.relu(conv3, name='conv3')

    # dropout
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    # Pooling. Input = 8x8x128. Output = 4x4x128.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten. Input = 4x4x128. Output = 4096 (?).
    fc0   = flatten(conv2)
    
    # Layer 4: Fully Connected. Input = 4096. Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(4096, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    # Layer 5: Fully Connected. Input = 1024. Output = 512.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(512))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Layer 6: Fully Connected. Input = 512. Output = 3 (n_classes).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return(logits)

def evaluate(X_data, y_data, accuracy_operation, x, y, keep_prob):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def display_image_predictions(features, labels, predictions, n_samples, n_predictions):
    n_classes = len(CLASSID_CLASSNAME.keys())
    print("n_classes="+str(n_classes))
#    label_names = signNames
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(figsize=(8, n_samples*1.5), nrows=n_samples, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [CLASSID_CLASSNAME[pred_i] for pred_i in pred_indicies]
        correct_name = CLASSID_CLASSNAME[label_id]
        
        # print the softmax values
        print("softmax predictions: ", pred_values[::-1])
        
        # reshape the image to be plottable by matplotlib
        feature_reshaped = feature.reshape((32, 32))
        
        axies[image_i][0].imshow(feature_reshaped, cmap="gray")
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
        
        # text annotations
        counter = 0
        for pred_value in sorted(pred_values[::-1]):
            pred_value_string = "{:05.3f}".format(pred_value)
            axies[image_i][1].text(pred_value + margin , counter, pred_value_string)
            counter += 1

    plt.show()
    plt.savefig(args.outdir + '/result_of_test_classifications.png')

        


def batch_features_labels(x, y, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        yield x[start:end], y[start:end]

def test_model(x, y, save_model_path, n_samples=4, top_n_predictions=3):
    """
    Test the saved model against the test dataset
    """

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in batch_features_labels(x, y, BATCH_SIZE):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(x, y)), n_samples)))
        
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        display_image_predictions(random_test_features, random_test_labels, random_test_predictions, n_samples, top_n_predictions)








TARGET_COUNT_PER_SIGN_TYPE = 1000
EPOCHS = 1
BATCH_SIZE = 128
dropout = 0.7

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
CLASSID_CLASSNAME = get_label_map(args.label_map)
n_classes = len(CLASSID_CLASSNAME.keys())

# load data
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(args.train_file, args.valid_file, args.test_file)

# augment training data
print("augmenting training data")
X_train, y_train = augment(X_train, y_train)

# preprocess data
print("features shape before preprocessing")
print("shape of training features: " + str(X_train.shape))
print("shape of testing features: " + str(X_test.shape))

print("preprocessing training data")
X_train, y_train = preprocess(X_train, y_train, n_classes)

print("preprocessing validation data")
X_valid, y_valid = preprocess(X_valid, y_valid, n_classes)

print("preprocessing testing data")
X_test, y_test = preprocess(X_test, y_test, n_classes)

print("----")
print("features shape after preprocessing")
print("shape of training features: " + str(X_train.shape))
print("shape of testing features: " + str(X_test.shape))

# shuffle training data
X_train, y_train = shuffle(X_train, y_train)

# check normalization
maxv = -10
minv = 10
avrg = 0

nr = random.randrange(0, len(X_train)-1, 1)
img = X_train[nr]

if np.amin(img) < minv:
    minv = np.amin(img)
if np.amax(img) > maxv:
    maxv = np.amax(img)

avrg = np.mean(img)

print('min value in random example image =', minv)
print('max value in random example image =', maxv)
print('mean of all values in a random example image =', avrg)


x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name='x')
y = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

rate = 0.0001

# get the logits and the convolutional layers for later visualization
logits = LeNet(x, keep_prob, n_classes)
# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
saver = tf.train.Saver()

init = tf.global_variables_initializer()

latest_checkpoint_file = None

# start training session
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    
    print(st + ": Training...")
    print(" ")

    for i in range(EPOCHS):
        #X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            
            
        validation_accuracy = evaluate(X_valid, y_valid, accuracy_operation, x, y, keep_prob)
        #print("Epoch {} Batch {} Validation Accuracy {:.3f}...".format(i+1, int(offset/BATCH_SIZE+1), validation_accuracy))
        
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(st + ": Epoch {} Validation Accuracy = {:.3f}".format(i+1, validation_accuracy))
 #       print()
    
        checkpoint_file = args.outdir + '/lenet.ckpt-'+str(i)
        saver.save(sess, checkpoint_file)
        print("Model saved as " + checkpoint_file)

        latest_checkpoint_file = checkpoint_file

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

# testing trained model
print("testing trained model")
test_model(X_test, y_test, latest_checkpoint_file)


#if __name__ == '__main__':
#    main()
