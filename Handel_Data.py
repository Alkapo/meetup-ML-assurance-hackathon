import math, json, os, sys
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

from random import shuffle

def randomize_files(file_list):
    shuffle(file_list)

from math import floor

def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing




def ml_function(datadir, num_folds):
    data_files = get_file_list_from_dir(datadir)
    randomize_files(data_files)
    for train_set, test_set in cross_validate(data_files, num_folds):
        do_ml_training(train_set)
        do_ml_testing(test_set)



import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE




######################################################################
##  2) Exploring the Data
# Load the dataset
path  = input('Veuillez rentrer le path ')
df = pd.read_csv(path)

df.rename(columns = {'green_roof_ind':'default'}, inplace=True)
df = df.drop(["Unnamed: 0"],axis=1)
df = df.drop(["ID"],axis=1)


'''
df = df.drop(["PERIODID_MY"],axis=1)
df  =df.iloc[:11900]
df = df.drop(df.columns[2:15],axis=1)
df = df.drop(df.columns[346:352],axis=1)
'''
df.head()


# Let's see if there are empty values.

print(df.isnull().sum())


#Create a new Class for Non Default observations.
df.loc[df.default == 0, 'nonDefault'] = 1
df.loc[df.default == 1, 'nonDefault'] = 0

print(df.default.value_counts())
print()
print(df.nonDefault.value_counts())

#Create dataframes of only default and nonDefault observations.
Default = df[df.default == 1]
NonDefault = df[df.nonDefault == 1]

# Set X_train equal to 80% of the observations that defaulted.
X_train = Default.sample(frac = 0.8)
count_Defaults = len(X_train)

# Add 80% of the not-defaulted observations to X_train.
X_train = pd.concat([X_train, NonDefault.sample(frac = 0.8)], axis = 0)

# X_test contains all the observations not in X_train.
#X_te = df.loc[~df.index.isin(X_train.index)]
X_test = df.loc[~df.index.isin(X_train.index)]




#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)


#Add our target classes to y_train and y_test.
y_train = X_train.default
y_train = pd.concat([y_train, X_train.nonDefault], axis=1)

y_test = X_test.default
y_test = pd.concat([y_test, X_test.nonDefault], axis=1)

#Drop target classes from X_train and X_test.
X_train = X_train.drop(['default','nonDefault'], axis = 1)
X_test = X_test.drop(['default','nonDefault'], axis = 1)

#Check to ensure all of the training/testing dataframes are of the correct length
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

# CHECKED !

#Names of all of the features in X_train.
features = X_train.columns.values


#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    '''if (df[feature].isnull().sum()> 0 ):
        df = df.drop([str(feature)], axis = 1)
    else:'''
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

#df.to_csv("C:/Users/ali_a/Desktop/AUT-2018/Intact/meetup-ML-assurance-hackathon/data/train_001.csv", sep='\t', encoding='utf-8')

# Split the testing data into validation and testing sets
split = int(len(y_test)/2)

inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]

# Number of input nodes.
input_nodes = 2048 #346

# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 5   #3

# Number of nodes in each hidden layer
hidden_nodes1 = input_nodes
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)

# Percent of nodes to keep during dropout.
pkeep = tf.placeholder(tf.float32)

# input
x = tf.placeholder(tf.float32, [None, input_nodes])

# layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.15))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.15))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

# layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.15))
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
y3 = tf.nn.dropout(y3, pkeep)

# layer 4
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev = 0.15))
b4 = tf.Variable(tf.zeros([2]))
y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

# output
y = y4
y_ = tf.placeholder(tf.float32, [None, 2])

# Parameters
training_epochs = 100     # These proved to be enough to let the network learn0
training_dropout = 0.
display_step = 2 # 10
n_samples = y_train.shape[0]
batch_size = 38
learning_rate = 0.001

# Cost function: Cross Entropy
cost = -tf.reduce_sum(y_ * tf.log(y))

# We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction if the most likely value (default or non Default) from softmax equals the target value.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###### Train the network
accuracy_summary = [] # Record accuracy values for plot
cost_summary = [] # Record cost values for plot
valid_accuracy_summary = []
valid_cost_summary = []
stop_early = 0 # To keep track of the number of epochs before early stopping

# Initialize variables and tensorflow session
saver = tf.train.Saver()

# Save the best weights so that they can be used to make the final predictions
#checkpoint = "C:/Users/ali_a/Desktop/Ete2018/CreditCardPrediction/final_mdl/best_model.ckpt"
#saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples/batch_size)):
            batch_x = inputX[batch*batch_size : (1+batch)*batch_size]
            batch_y = inputY[batch*batch_size : (1+batch)*batch_size]

            sess.run([optimizer], feed_dict={x: batch_x,
                                             y_: batch_y,
                                             pkeep: training_dropout})

        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX,
                                                                            y_: inputY,
                                                                            pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={x: inputX_valid,
                                                                                  y_: inputY_valid,
                                                                                  pkeep: 1})

            print ("Epoch:", epoch,
                   "Acc =", "{:.5f}".format(train_accuracy),
                   "Cost =", "{:.5f}".format(newCost),
                   "Valid_Acc =", "{:.5f}".format(valid_accuracy),
                   "Valid_Cost = ", "{:.5f}".format(valid_newCost))

            # Record the results of the model
            accuracy_summary.append(train_accuracy)
            cost_summary.append(newCost)
            valid_accuracy_summary.append(valid_accuracy)
            valid_cost_summary.append(valid_newCost)

            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_summary) and epoch > 130000:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0
    saver.save(sess, './Model/my_test_model_15_02')
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    tf.saved_model.simple_save(sess,"./Model/saved_model")
'''
with tf.Session() as sess:
    # Load the best weights and show its results
    saver.restore(sess, checkpoint)
    training_accuracy = sess.run(accuracy, feed_dict={x: inputX, y_: inputY, pkeep: training_dropout})
    validation_accuracy = sess.run(accuracy, feed_dict={x: inputX_valid, y_: inputY_valid, pkeep: 1})
    
    print("Results using the best Valid_Acc:")
    print()
    print("Training Accuracy =", training_accuracy)
    print("Validation Accuracy =", validation_accuracy)
'''

print()
print('the max valid in sumary is  :', np.amax(valid_accuracy_summary))
print()

#Plot accuracy and cost summary

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

ax1.plot(accuracy_summary)
ax1.set_title('Accuracy')

ax2.plot(cost_summary)
ax2.set_title('Cost')

plt.xlabel('Epochs (x100)')
plt.show()

### This section is to predict new value to the competition test file :


def TB(cleanup = False):
    '''import webbrowser
    #webbrowser.open('127.0.1.1:6006')
    tf.flags.FLAGS.lo
    tb.main()'''
    pa = './graphs/'
    import webbrowser
    webbrowser.Chrome('http://127.0.1.1:6006')
    os.system('tensorboard --logdir=' + pa)
TB(1)


# Load the dataset

d2 = pd.read_csv("C:/Users/ali_a/Desktop/Ete2018/DGAG/CreditCardPrediction/EvalData/eval_ali_104.csv")
d2.rename(columns = {'Default':'default'}, inplace=True)
d2 = d2.drop(["PERIODID_MY"],axis=1)
d2.head()

y_pred = d2.drop(['default'], axis = 1)
print(len(y_pred))


# Let's see if there are empty values.
y_pred.isnull().sum()

#Names of all of the features in X_train.
features = y_pred.columns.values

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    mean, std = d2[feature].mean(), d2[feature].std()
    y_pred.loc[:, feature] = (d2[feature] - mean) / std




with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./Model/my_test_model_15_02.meta')
    print("Model restored.")

    saver.restore(sess, tf.train.latest_checkpoint('./Model/'))
    print("Model restored   compl.")

    y_pred = y_pred.as_matrix()

    output = sess.run(y,feed_dict={x: y_pred, pkeep:1})

    print("Output is predicted to be ")
    df1001 = pd.DataFrame(output)
    df1001.to_csv("C:/Users/ali_a/Desktop/Ete2018/CreditCardPrediction/logOut/AliBabaAnd40Hackers_15_02.csv", sep='\t', encoding='utf-8')
#predict_fn = predictor.from_saved_model("C:/Users/ali_a/Desktop/Ete2018/CreditCardPrediction/")




print("Optimization Finished!")
print()



# To summarize the confusion matrix:


#pandas.DataFrame(y_predicted).to_csv("C:/Users/ali_a/Desktop/Ete2018/CreditCardPrediction/AliBabaAnd40Hackers104.csv", sep='\t', encoding='utf-8')

#tf.pre.to_csv("C:/Users/ali_a/Desktop/Ete2018/CreditCardPrediction/AliBabaAnd40Hackers.csv", sep='\t', encoding='utf-8')

