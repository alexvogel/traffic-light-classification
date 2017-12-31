# from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

import argparse
import tensorflow as tf
import cv2
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    
    return graph

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

def preprocess(x):
    '''
    : x: features vector
    : y: labels vector
    '''
    x = rgb2gray(x)

    x = normalize(x)
    
    x = reshape(x)

    
    return(x.astype(np.float32))


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--image", default="", type=str, help="Image to classify")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)

        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    loaded_x = graph.get_tensor_by_name('prefix/x:0')
    loaded_y = graph.get_tensor_by_name('prefix/y:0')
    loaded_keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
    loaded_logits = graph.get_tensor_by_name('prefix/logits:0')
    loaded_acc = graph.get_tensor_by_name('prefix/accuracy:0')
    
    # read image
    batch = []
    img = cv2.imread(args.image)
    batch.append(img)
    batch = np.asarray(batch)

    batch = preprocess(batch)

    # check normalization
    maxv = -10
    minv = 10
    avrg = 0

    # there is only 1 image
    nr = 0
    tmp_img = batch[nr]

    if np.amin(tmp_img) < minv:
        minv = np.amin(tmp_img)
    if np.amax(tmp_img) > maxv:
        maxv = np.amax(tmp_img)

    avrg = np.mean(tmp_img)

    print('min value in random example image =', minv)
    print('max value in random example image =', maxv)
    print('mean of all values in a random example image =', avrg)


    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        logits = sess.run(loaded_logits, feed_dict={loaded_x: batch, loaded_y: np.ndarray(shape=(1,3)), loaded_keep_prob: 1.0})
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
    print(logits) # [[ False ]] Yay, it works!
