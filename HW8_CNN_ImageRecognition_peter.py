
# coding: utf-8

# In[1]:


#HW8 - Image recognition with inception_v3 
#HW8.0 Prepare the environment, create dataset and image folders, find and save an image in .jpg format
#HW8.1 Image loading and manipulation
#HW8.2 Fetching and loading inception_v3 model
#HW8.3 Recognizing My_Dog1.png image
#HW8.4 Recognizing your own image


# In[145]:


#HW8.0. I recommend that you create a new folder for this HW, e.g. HW8, and make it a root directory 
#Save this file in your HW8 folder
#inside HW8 folder create two subfolders: datasets and images
#Download and save the images myDog_raw.jpg. Images myDog_raw.png, myDog_ready.png are provided just in case.
#Download imagenet_class_labels.txt file and place it in you HW8 folder (you will need to move it later)
#Run the code below to set up your python environment


# In[152]:


print ('Oluwaseyitan Awojobi \nBCIS 5690')


# In[110]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "Desktop/hw8/images"
#CHAPTER_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image):
    plt.imshow(images, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(images.astype(np.uint8),interpolation="nearest")
    plt.axis("off")


# In[111]:


#HW8.1. Image loading and manipulation


# In[113]:


#convert .jpg image to .png if you do not have a .png image
from PIL import Image

im = Image.open(os.path.join("Desktop/hw8/images", "fines.jpeg"))
im.save(os.path.join("Desktop/hw8/images","fines1.png"), "PNG")


# In[114]:


#importing raw image and displaying it
import matplotlib.image as mpimg
test_image = mpimg.imread(os.path.join("Desktop/hw8/images","fines1.png"))
plt.imshow(test_image)
plt.axis("off")
plt.show()


# In[116]:


#cropping the image to a  centered square, then resizing it  
from PIL import Image

im = Image.open(os.path.join("Desktop/hw8/images","fines1.png"))

def crop_resize_image(im, width, height):
    x, y = im.size
    crop_size = min(x, y)
    x_marg=max(0, (x - crop_size)/2)
    y_marg=max(0, (y - crop_size)/2)
    crop_im = im.crop((x_marg, y_marg, x_marg+crop_size, y_marg+crop_size))
    new_im=crop_im.resize((width, height), Image.ANTIALIAS)
    new_im.save(os.path.join("Desktop/hw8/images","fines2.png"), "PNG")
    return new_im

#defining width, height and channels
width = 299
height = 299
channels = 3

#performing the actual croppint and resizing, displaying the image
new_im=crop_resize_image(im, width, height)
plt.imshow(new_im)
plt.axis("off")
plt.show()


# In[117]:


#opening ready image in matplotlib
import matplotlib.image as mpimg
test_image = mpimg.imread(os.path.join("Desktop/hw8/images","fines2.png"))
plt.imshow(test_image)
plt.axis("off")
plt.show()


# In[118]:


#adjusting the format of the image to fit inception 3 model
test_image = 2 * test_image - 1
plt.imshow(test_image)
plt.axis("off")
plt.show()


# In[120]:


#HW8.2 Loading Inception v.3 model


# In[128]:


#defining the parameters for fetching inception v.3 model
import sys
import tarfile
from six.moves import urllib

TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("Desktop/hw8/datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "inception_v3.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)


# In[129]:


#actually fetching pretrained inception_v3 model
fetch_pretrained_inception_v3()


# In[130]:


#ATTENTION: here you need to copy the file imagenet_class_labels.txt into datasets/inception folder


# In[135]:


#Compiling the model, loading class names 
import re

CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)

def load_class_names():
    with open(os.path.join("Desktop/hw8/datasets", "inception", "imagenet_class_names.txt"), "rb") as f:
        content = f.read().decode("utf-8")
        return CLASS_NAME_REGEX.findall(content)


# In[137]:


#loading class names and displaying top 5 class names
class_names = ["background"] + load_class_names()
class_names[:5]


# In[138]:


#defining tensorflow graph to make predictions using inception_v3 model

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()


# In[139]:


#HW8.3. Perform image recognition on fines2.png


# In[141]:


#reloading the right image one more time, just in case
import matplotlib.image as mpimg
test_image = mpimg.imread(os.path.join("Desktop/hw8/images","fines2.png"))
print(test_image.shape)
test_image = 2 * test_image - 1
plt.imshow(test_image)
plt.axis("off")
plt.show()


# In[142]:


#making predictions for a specific image
import tensorflow as tf

X_test = test_image.reshape(-1, height, width, channels)

with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    predictions_val = predictions.eval(feed_dict={X: X_test})


# In[143]:


#displaying most likely image class

most_likely_class_index = np.argmax(predictions_val[0])
most_likely_class_index
class_names[most_likely_class_index]


# In[144]:


#displaying top 5 most likely image classes with corresponding probabilities

top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])
for i in top_5:
    print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))

