
# coding: utf-8

# In[ ]:


#Code source https://github.com/ageron/handson-ml/


# In[242]:


print ('Name: Oluwaseyitan Awojobi.\nI made edits as necessary and also tried the code on a different dataset. \nOverall result is great')


# In[ ]:


#HW7.0. This is mostly the repetition of the code from HW3, but numeric features are modified
#specifically, missing values are imputed and featured are scaled
#finally, the resulting prepared dataset is saved in a .csv file


# In[157]:


#importing pandas and reading .csv file, dropping customerID column, displaying the head

import pandas as pd
data=pd.read_csv("OneDrive/FALL 2018/BCIS 5690/Telcos.csv")
data=data.drop(["customerID"], axis=1)
data.head()


# In[160]:


# displaying data type info for all columns

data.info()


# In[161]:


#converting TotalCharges to float data type

data["TotalCharges"]=pd.to_numeric(data["TotalCharges"], errors='coerce')
data["TotalCharges"].dtype


# In[162]:


#converting SeniorCitizen to a categorical column for future convenience

data["SeniorCitizen"]=data["SeniorCitizen"].astype(str)
data["SeniorCitizen"].dtype


# In[163]:


#getting descriptive statistics on numeric columns

data.describe()


# In[164]:


#visualising numeric data using a histogram

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=20, figsize=(20,15))


# In[165]:


#P3.2. Dealing with categorical variables 
#Display value counts for InternetService and SeniorCitizen variables
#Convert categorical variables (those with the data type object) to numerical using LabelEncoder class from sklearn.preprocessing
#Convert variables that have more than 2 categories to one-hot vectors using OneHotEncoder class from sklearn.preprocessing
#Combine numeric variables, encoded variables with two categories, and variables encoded into one-hot arrays into one data frame. 
#Creating a data ndarray from the df_complete dataframe


# In[166]:


#spliting data columns into numeric and categorical

df_num_counter=0
df_other_counter=0
for col in data:
    if (data[col].dtype)in ["int64", "float64"]:
        df_num_col=pd.DataFrame(data[col], columns=[col])
        if df_num_counter==0:
            df_num=df_num_col
        else:
            df_num=df_num.join(df_num_col)
        df_num_counter=df_num_counter+1
    else:
        df_other_col=pd.DataFrame(data[col], columns=[col])
        if df_other_counter==0:
            df_other=df_other_col
        else:
            df_other=df_other.join(df_other_col)
        df_other_counter=df_other_counter+1  


# In[167]:


#displaying numeric columns data frame
df_num.head()


# In[168]:


col_names=list(df_num)
col_names


# In[169]:


#imputing missing values
#scaling numeric data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

imputer= Imputer(strategy="median")
scaler=StandardScaler()

col_names=list(df_num)
num_col=np.array(df_num)
num_col_imp=imputer.fit_transform(num_col)
num_col_scaled=scaler.fit_transform(num_col_imp)

num_col_scaled

df_num_scaled=pd.DataFrame(num_col_scaled, columns=col_names)

df_num_scaled.head()


# In[170]:


#displaying non-numeric columns data frame
df_other.head()


# In[171]:


#encoding non_numeric columns


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
i=0
for col in df_other:
    x_enc=encoder.fit_transform(df_other[col])
    df_enc=pd.DataFrame(x_enc, columns=[col])
    if i>0:
        df_cat=df_cat.join(df_enc)
    else:
        df_cat=df_enc
    i=i+1
    #print(col, " ", encoder.classes_)
df_cat.head()


# In[172]:


#splitting encoded columns into binary and multi-category data frames

bin_count=0
mult_count=0

for col in df_cat:
    #print(df_cat[col].value_counts())
    if len(df_cat[col].unique())>2:
        #print (col, "has ", len(df_cat[col].unique()), " unique values")
        mult_col=df_cat[col]
        if mult_count==0:
            df_mult=pd.DataFrame(mult_col, columns=[col])
        else:
            df_mult=df_mult.join(mult_col)
        mult_count=mult_count+1
    else:
        bin_col=df_cat[col]
        if bin_count==0:
            df_bin=pd.DataFrame(bin_col, columns=[col])
        else:
            df_bin=df_bin.join(bin_col)
        bin_count=bin_count+1


# In[173]:


#displaying multiple category columns
df_mult.head()


# In[174]:


#displaying binary columns
df_bin.head()


# In[175]:


#converting multi-category columns to one-hot-arrays using loops

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()

df_counter=0
for col in df_mult:

    v_len=len(df_mult[col].unique())
    col_names=[]
    i=0
    col_pref=col
    while i<v_len:
            i=i+1
            col_name=col_pref+str(i)
            col_names.append(col_name)
    #print (col_names)    
    next_cols_hot1_enc=encoder.fit_transform(df_mult[col].values.reshape(-1,1))
    next_cols=next_cols_hot1_enc.toarray()
    df_next_cols=pd.DataFrame(next_cols, columns=[col_names])
    
    if df_counter==0:
        df_mult_one_hot=df_next_cols
    else:
        df_mult_one_hot=df_mult_one_hot.join(df_next_cols)
    df_counter=df_counter+1   
    
df_mult_one_hot.head()


# In[176]:


#combinig one_hot encoded and binary columns

df_enc_complete=df_bin.join(df_mult_one_hot)
df_enc_complete.head()


# In[177]:


#cobmining scaled numeric columns with binary and one-hot-encoded

df_complete=df_num_scaled.join(df_enc_complete)
df_complete.head()


# In[178]:


#saving data to a file
df_complete.to_csv('OneDrive/FALL 2018/BCIS 5690/Telcos_ready.csv')


# In[179]:


#HW7.1 Opening Telcos_ready.csv dataset, splitting it into features and labels, splitting it into train and test
#if you have downloaded Telcos_ready dataset from Blackboard, you can start from here. 


# In[180]:


import pandas as pd
import numpy as np
df_complete=pd.read_csv('OneDrive/FALL 2018/BCIS 5690/Telcos_ready.csv')

df_telcos_features=df_complete.drop("Churn", axis=1)
df_telcos_features.info()


# In[181]:


telcos_labels=df_complete["Churn"]
df_telcos_labels=pd.DataFrame(telcos_labels, columns=["Churn"])
df_telcos_labels.info()


# In[182]:


#creating numpy arrays
X=np.array(df_telcos_features)
y=np.array(df_telcos_labels)


# In[183]:


#creating test and train datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#getting the shape of X_train array
n, m=X_train.shape
n, m


# In[184]:


#H7.2. Fitting logistic regression, assessing model accurasy vis-a-vis baseline classifier 


# In[185]:


#Import and fit LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train.ravel())


# In[186]:


#assessing model accuracy for train dataset using cross-validation
# for your HW submission increase the number of folds to 10 
#by changing the cv parameter

y_train_pred=log_reg.predict(X_train)
y_test_pred=log_reg.predict(X_test)
from sklearn.model_selection import cross_val_score
cross_val_score(log_reg, X_train, y_train.ravel(), cv=10, scoring="accuracy")


# In[187]:


#assessing model accuracy for train dataset using cross-validation
y_test_pred=log_reg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test_pred, y_test.ravel())


# In[188]:


y_train_pred=log_reg.predict(X_train)
n_correct=sum(y_train_pred==y_train.ravel())
n_correct
print("Testing Accuracy: ", n_correct/len(y_train_pred))


# In[189]:


#what about the accuracy of a classifier that will simply predict no churn?

from sklearn.base import BaseEstimator

class NeverChurnClassifier(BaseEstimator):
    def fit (self, y=None):
        pass
    def predict (self, X):
        return np.zeros((len(X), 1), dtype="int64")
    


# In[190]:


nccf=NeverChurnClassifier()
y_train_nccf_pred=nccf.predict(X_train)
n_correct=sum(y_train_nccf_pred==y_train)
n_correct
print("Baseline Accuracy: ", n_correct/len(y_train_pred))


# In[191]:


#HW7.3 Using high level tensorflow API to fit a neural network


# In[192]:



import tensorflow as tf
tf.reset_default_graph()

n,m=X_train.shape

X_train = X_train.astype(np.float32).reshape(-1, m) 
X_test = X_test.astype(np.float32).reshape(-1, m) 
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:1500], X_train[1500:]
y_valid, y_train = y_train[:1500], y_train[1500:]
len(X_train)


# In[193]:


#training DNN using high level APIs
#for your HW, change the number of epochs to 75, 
#for your HW, change the number of hidden layers to 2 with 20 and 10 neurons respoctively. 
#have only 2 hidden layers with 

feature_cols = [tf.feature_column.numeric_column("X", shape=[m])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[2,20], n_classes=2,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=75, batch_size=75, shuffle=True)
dnn_clf.train(input_fn=input_fn)


# In[194]:


#evaluating the model using high level APIs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)


# In[198]:


#HW7.4. Download MNIST dataset and fit a neural network with 2 hidden layers, 300, 200 and 100 neurons each
#All of the code is given, but you will need to change parameters.


# In[ ]:


#Fitting DNN on MNIST data set using high-level API
import tensorflow as tf

tf.reset_default_graph()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


# In[ ]:


#change the number of hidden layers to 2, with 200 and 100 units respectively.
#compare the accuracy score with the solution provided.

feature_cols = [tf.feature_column.numeric_column("X", shape=[28*28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 200, 100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=50, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)


# In[210]:


#evaluating the model using high level APIs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)


# In[211]:


#HW7.5. THis is the last part, I promise!
#Defining and training a neural network using basic Tensorflow
#Here, all the code is given, but you will need to increase the number 
#of epochs to 50


# In[212]:


import tensorflow as tf
tf.reset_default_graph()

m, n=X_train.shape

n_inputs = n
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
n_epochs=50
batch_size=50

m, n


# In[213]:


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


# In[215]:


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


# In[216]:


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")


# In[217]:


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


# In[218]:


learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[219]:


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# In[221]:


init = tf.global_variables_initializer()
#saver = tf.train.Saver()


# In[222]:


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        


# In[223]:


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    #save_path = saver.save(sess, "./my_model_final.ckpt")

