#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 09:16:14 2017

@author: yangliu
"""

import pandas as pd 
import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt


#Define a function to show image through 48*48 pixels
def show(img):
    show_image = img.reshape(48,48)
    
    #plt.imshow(show_image, cmap=cm.binary)
    plt.imshow(show_image, cmap='gray')


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def data_augmentation(old_images, old_labels):
    new_images = []
    new_labels = []
    for i in range(len(old_images)):
        new_images.append(old_images[i])
        img = old_images[i]
        img = np.reshape(img, (48,48))
        img = np.flip(img, axis=1)
        img = np.reshape(img, 48*48)
        new_images.append(img)
        new_labels.append(old_labels[i])
        new_labels.append(old_labels[i])
    new_images = np.reshape(new_images, (-1, 48*48))
    new_labels = np.reshape(new_labels, (-1, 7))
    return new_images, new_labels

def read_data(file):
    data = pd.read_csv(file)
    print(data.shape)
    print(data.head())
    print(np.unique(data["Usage"].values.ravel()))
    print ('The number of training data set is %d'%(len(data[data.Usage == "Training"])))
    train_data = data[data.Usage == "Training"]
    pixels_values = train_data.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.float)
    print(images)
    
    #show one image
    show(images[8])
    
    images = images - images.mean(axis=1).reshape(-1,1)
    images = np.multiply(images,100.0/255.0)
    each_pixel_mean = images.mean(axis=0)
    each_pixel_std = np.std(images, axis=0)
    images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)
    print('images.shape: ' + str(images.shape))
    image_pixels = images.shape[1]
    print('Flat pixel values is %d'%(image_pixels))
    image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)
    print('image_width: ' + str(image_width))
    print('image_height: ' + str(image_height))
    labels_flat = train_data["emotion"].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    print(np.unique(labels_flat))
    print('The number of different facial expressions is %d'%labels_count)
    
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    
    print('labels[0]: ' + str(labels[0]))
    # split data into training & validation
    VALIDATION_SIZE = 1709
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    
    #train_images = images[VALIDATION_SIZE:]
    #train_labels = labels[VALIDATION_SIZE:]
    train_images, train_labels = data_augmentation(images[VALIDATION_SIZE:], labels[VALIDATION_SIZE:])
    print ('The number of final training data: %d'%(len(train_images)))
    
    return data, train_images, train_labels, validation_images, validation_labels, each_pixel_mean, each_pixel_std
    
 
# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    num_examples = train_images.shape[0]
    start = index_in_epoch
    index_in_epoch += batch_size
    epochs_completed = 0
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]



def graph():
    # input & output of NN
    
    # images
    x = tf.placeholder('float', shape=[None, image_pixels])
    # labels
    y_ = tf.placeholder('float', shape=[None, labels_count])
    
    # first convolutional layer 64
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    
    # (27000, 2304) => (27000,48,48,1)
    image = tf.reshape(x, [-1,image_width , image_height,1])
    #print (image.get_shape()) # =>(27000,48,48,1)
    
    
    h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
    #print (h_conv1.get_shape()) # => (27000,48,48,64)
    h_pool1 = max_pool_2x2(h_conv1)
    #print (h_pool1.get_shape()) # => (27000,24,24,1)
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])
    
    h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
    #print (h_conv2.get_shape()) # => (27000,24,24,128)
    
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    h_pool2 = max_pool_2x2(h_norm2)
    
    # local layer weight initialization
    def local_weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.04)
        return tf.Variable(initial)
    
    def local_bias_variable(shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)
    
    # densely connected layer local 3
    W_fc1 = local_weight_variable([12 * 12 * 128, 3072])
    b_fc1 = local_bias_variable([3072])
    
    # (27000, 12, 12, 128) => (27000, 12 * 12 * 128)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #print (h_fc1.get_shape()) # => (27000, 1024)
    
    # densely connected layer local 4
    W_fc2 = local_weight_variable([3072, 1536])
    b_fc2 = local_bias_variable([1536])
    
    # (40000, 7, 7, 64) => (40000, 3136)
    h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)
    #print (h_fc1.get_shape()) # => (40000, 1024)
    # dropout
    keep_prob = tf.placeholder('float')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    # readout layer for deep net
    W_fc3 = weight_variable([1536, labels_count])
    b_fc3 = bias_variable([labels_count])
    
    y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    print ('y.get_shape()' + str(y.get_shape())) # => (40000, 10)
    
    # settings
    LEARNING_RATE = 1e-4
    
    # cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    
    
    # optimisation function
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cross_entropy)
    
    # evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    # prediction function
    #[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
    predict = tf.argmax(y,1)
    predict_softmax = tf.nn.softmax(y)

    
    return x, y_, predict, predict_softmax, keep_prob, train_step, accuracy
    
    
def train():    
    # set to 3000 iterations 
    TRAINING_ITERATIONS = 5000
        
    DROPOUT = 0.5
    
    # start TensorFlow session
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    
    sess.run(init)
    
    # visualisation variables
    # global train_accuracies
    # global validation_accuracies
    # global x_range
    
    display_step=1        
    
    for i in range(TRAINING_ITERATIONS):
        #get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    
        # check progress on every 1st,2nd,...,10th,20th,...,100th... step
        if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})       
            if(VALIDATION_SIZE):
                validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], y_: validation_labels[0:BATCH_SIZE], keep_prob: 1.0})                                  
                print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))                
                validation_accuracies.append(validation_accuracy)               
            else:
                print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
            train_accuracies.append(train_accuracy)
            x_range.append(i)
            
            # increase display_step
            if i%(display_step*10) == 0 and i and display_step<100:
                display_step *= 10
        # train on batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
        
    return sess, x_range, accuracy, train_accuracies, validation_accuracies



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
        
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
def analysis(train_accuracies, validation_accuracies, x_range):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    print ('len(train_accuracies)' + str(len(train_accuracies)))
    print ('len(validation_accuracies)' + str(len(validation_accuracies)))
    print ('len(x_range)' + str(len(x_range)))
    print ('train_accuracies = ' + str(train_accuracies))
    print ('validation_accuracies = ' + str(validation_accuracies))
    print ('x_range = ' + str(x_range))

    # check final accuracy on validation set  
    if(VALIDATION_SIZE):
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                       y_: validation_labels, 
                                                       keep_prob: 1.0})
        print('validation_accuracy => %.4f'%validation_accuracy)
        plt.plot(x_range, train_accuracies,'-b', label='Training')
        plt.plot(x_range, validation_accuracies,'-g', label='Validation')
        plt.legend(loc='lower right', frameon=False)
        plt.ylim(ymax = 1.0, ymin = 0.0)
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.show()
        
    saver = tf.train.Saver(tf.all_variables())   
    #saver.save(sess, 'my-model1', global_step=0)
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)   
    
    # read test data from CSV file 
    test_data = data[data.Usage == "PublicTest"]   
    print('test_data.head(): ' + str(test_data.head()))
    print('len(test_data): ' + str(len(test_data)))
    
    test_pixels_values = test_data.pixels.str.split(" ").tolist()
    test_pixels_values = pd.DataFrame(test_pixels_values, dtype=int)
    test_images = test_pixels_values.values
    test_images = test_images.astype(np.float)
    test_images = test_images - test_images.mean(axis=1).reshape(-1,1)
    test_images = np.multiply(test_images,100.0/255.0)
    test_images = np.divide(np.subtract(test_images,each_pixel_mean), each_pixel_std)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    print('each_pixel_mean: ' + str(each_pixel_mean))
    print('each_pixel_std: ' + str(each_pixel_std))

    # predict test set
    #predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})
    
    # using batches is more resource efficient
    predicted_lables = np.zeros(test_images.shape[0])
        
    for i in range(0,test_images.shape[0]//BATCH_SIZE):
        predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                    keep_prob: 1.0})
    
    
    print('predicted_lables({0})'.format(len(predicted_lables)))
    
    print('predicted_lables: ' + str(predicted_lables))
    print('test_data.emotion.values: ' + str(test_data.emotion.values))
    print('accuracy_score(test_data.emotion.values, predicted_lables): ' + str(accuracy_score(test_data.emotion.values, predicted_lables)))
    print('confusion_matrix(test_data.emotion.values, predicted_lables): ' + str(confusion_matrix(test_data.emotion.values, predicted_lables)))
        
    cnf_matrix = confusion_matrix(test_data.emotion.values, predicted_lables)
    
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion Matrix for Test Dataset')

    plt.show()
    
    
    
def test():
    import cv2
    
    emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6: 'Neutral'}
    #emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}
    face_path = '/usr/local/Cellar/opencv/2.4.13.2/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_path)
    #img = cv2.imread('image0.jpg')
    img = cv2.imread('image07.jpg')
    #img = cv2.imread('surprise.jpeg')
    #img = cv2.imread('angry.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
        
    with tf.Session() as sess:  
        saver.restore(sess, "model.ckpt")
        print("Model restored.")
    
        # Access saved Variables directly
        #print("W_conv1 : %s" % str(W_conv1))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for face in faces:
            (xx,yy,ww,hh) = face
            crop_img = gray[yy:yy+hh, xx:xx+ww]
            resized = cv2.resize(crop_img, (48, 48))
            test_image = np.reshape(resized.astype(np.float), 48*48)
            test_image = test_image - test_image.mean()
            test_image = np.multiply(test_image,100.0/255.0)
            print(test_image)
            print(len(test_image))
            print(test_image.dtype)
            
            each_pixel_mean = np.mean(test_image)
            each_pixel_std = np.std(test_image)
            test_image = np.divide(np.subtract(test_image,each_pixel_mean), each_pixel_std)
            print(test_image)
               
            predicted_lable = predict.eval(feed_dict={x: [test_image], keep_prob: 1.0})
            print("predicted_lable: " + str(predicted_lable))
            predicted_softmax = predict_softmax.eval(feed_dict={x: [test_image], keep_prob: 1.0})
            print("predicted_softmax: " + str(predicted_softmax))
            expression = emotion[predicted_lable[0]]
            print(expression)
            
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(img,(xx,yy),(xx+ww,yy+hh),(255,0,0),0)
            cv2.putText(img,expression, (xx,yy), font, 0.5, 255)

        while(1):
            cv2.imshow('image', img)
        
            # This is where we get the keyboard input
            # Then check if it's "m" (if so, toggle the drawing mode)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':

    model_path = 'model.ckpt.meta'
    image_pixels = 2304
    labels_count = 7
    image_width = 48
    image_height = 48
    VALIDATION_SIZE = 1709
    BATCH_SIZE = 50
    index_in_epoch = 0
    
    if (os.path.isfile(model_path)):
        x, y_, predict, predict_softmax, keep_prob, train_step, accuracy = graph()
        test()
        
    else:    
        file = 'fer2013/fer2013.csv'
        data, train_images, train_labels, validation_images, validation_labels, each_pixel_mean, each_pixel_std = read_data(file)
        

        
        train_accuracies = []
        validation_accuracies = []
        x_range = []
        
        x, y_, predict, predict_softmax, keep_prob, train_step, accuracy = graph()
        
        sess, x_range, accuracy, train_accuracies, validation_accuracies = train()
        
        analysis(train_accuracies, validation_accuracies, x_range)
    
