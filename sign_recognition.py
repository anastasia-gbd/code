import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from random import seed
from random import randrange
import numpy
import csv
import random

image_size = (30, 30)
n_est = 10

def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

#read train data(images and labels)
trainImages, trainLabels = readTrafficSigns('GTSRB-2/Final_Training/Images')

#Image padding
for i, elem in enumerate(trainImages):
    h, w, d = elem.shape
    new_size = max(h, w)
    img = Image.fromarray(elem, 'RGB')
    delta_h = new_size - h
    delta_w = new_size - w
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(img, padding)
    trainImages[i] = numpy.array(new_im)

#Image resize
for i, elem in enumerate(trainImages):
    size = image_size
    img = Image.fromarray(elem, 'RGB')
    trainImages[i] = numpy.array(ImageOps.fit(img, size, Image.ANTIALIAS))


# Split a dataset into a train and test set
train_images = []
test_images = trainImages
train_labels = []
test_labels = trainLabels
    
train_size = 0.8 * len(trainImages)
for_shuffle = []

#Take random images and form train data
while len(for_shuffle)*30 < train_size:
    index = randrange(len(test_images)//30)
    subarray = []
    for k in range(30):
        subarray.append([test_images.pop(index*30), test_labels.pop(index*30)])
    for_shuffle.append(subarray)
#Shuffle data
random.shuffle(for_shuffle)
for i in for_shuffle:
    for k in i:
        train_images.append(k[0])
        train_labels.append(k[1])

        
#Finding frequencies
set_of_labels = train_labels
#set_of_labels = trainLabels
dictionary = {}
for i in set_of_labels:
    if(i in dictionary.keys() ):
        dictionary[i] += 1
    else:
        dictionary[i] = 1
#Plot frequencies    
plt.figure(figsize=(10,5))
plt.ylabel('Number of images')
plt.xlabel('Class')
plt.bar(list(dictionary.keys()), dictionary.values(), width = 0.8, color="black")
plt.show()

from skimage import exposure
from skimage.util import random_noise
from skimage import transform
from cv2 import resize
import matplotlib.image as mpimg


#image augmentation
def augment_image(img):
    #choose random augmentation
    i = random.randint(0, 1) 
    if (i==1):
        return (random_noise(img, mode='s&p', clip=True))
    else:
        return (transform.rotate(img, random.uniform(-10,10)))

#create dictionary of different classes and images - makes implementation of augmentation easier
dictionary_of_classes = {}
for i, element in enumerate(set_of_labels):
    if(element in dictionary_of_classes.keys()):
        dictionary_of_classes[element].append(train_images[i])
    else:
        dictionary_of_classes[element] = [train_images[i]]
        

#Add augmented images to classes
for i in dictionary.keys():
    #If in class there are less images that max from all classes - add augmented
    if (dictionary[i]<max(dictionary.values())):
        class_size = len(dictionary_of_classes[i])
        while len(dictionary_of_classes[i]) < max(dictionary.values()):
            for k in range(30):
                img = random.choice(dictionary_of_classes[i][0:class_size])
                dictionary_of_classes[i].append(augment_image(img))
                

#Shaffle images in classes again
for_shuffle = []
train_images = []
train_labels = []
for i in dictionary_of_classes.keys():
    for index in range(len(dictionary_of_classes[i])//30 ):
        for_shuffle.append([i, [ dictionary_of_classes[i][(index*30):((index+1)*30)] ]  ])
random.shuffle(for_shuffle)

#form train images and labels
for i in for_shuffle:
    for k in i[1][0]:
        train_images.append(k)
        train_labels.append(i[0])
        
#Finding frequencies
set_of_labels = train_labels
#set_of_labels = trainLabels
dictionary = {}
for i in set_of_labels:
    if(i in dictionary.keys() ):
        dictionary[i] += 1
    else:
        dictionary[i] = 1
#Now all classes of images are equal    
plt.bar(list(dictionary.keys()), dictionary.values(), width = 0.8, color="black")


image_to_normalize = train_images

import cv2 as cv
#image normalization
def normalize_image(array):
    normalizedImg = numpy.zeros(image_size)
    normalizedImg = cv.normalize(array, normalizedImg, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return normalizedImg

#normalize image
for i, element in enumerate(image_to_normalize):
    image_to_normalize[i] = normalize_image(element)
for_making_vector = image_to_normalize

#make from image vector
def make_vector(array):
    images = []
    for i in array:
        images.append(i.flatten())
    return numpy.array(images)

#make from image one vector - for whole training data
for_making_vector = make_vector(for_making_vector)


#Create validation_set
validation_images = test_images
validation_labels = test_labels

#normalize validation set
for i, element in enumerate(validation_images):
    validation_images[i] = normalize_image(element)

#make vector for validation images
validation_images = make_vector(validation_images)

from sklearn.ensemble import RandomForestClassifier
import time
#Create Random Forest classifier
clf=RandomForestClassifier(n_estimators=n_est)

start_fit = time.time()
#train model
clf.fit(for_making_vector,train_labels)
finish_fit = time.time()

start_predict = time.time()
#predict for validation
y_pred=clf.predict(validation_images)
finish_predict = time.time()

print("Time difference",finish_fit-start_fit, finish_predict - start_predict )
from sklearn import metrics
#Calculate accuracy of validation set
print("Accuracy:", metrics.accuracy_score(validation_labels, y_pred))

#create confusion matrix for validation set
y_test = validation_labels
pred = y_pred

labels = []
for i in range(43):
    labels.append(str(i))

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, pred)
import pandas as pd
import seaborn as sns

df_cm = pd.DataFrame(conf_matrix, index = labels, columns = labels)
sns.heatmap(df_cm, cmap = "Greys", annot = True, fmt = 'g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#Read Test Data
images_final_test = [] # images
labels_final_test = [] # corresponding labels

image_for_printing_final = []

gtFile = open('GT-final_test.csv') # annotations file
gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
next(gtReader) # skip header
# loop over all images in current annotations file
for row in gtReader:
    images_final_test.append(plt.imread("GTSRB/Final_Test/Images/"+ row[0])) # the 1th column is the filename
    labels_final_test.append(row[7]) # the 8th column is the label
    image_for_printing_final.append([row[0], numpy.array(plt.imread("GTSRB/Final_Test/Images/"+ row[0]))])
gtFile.close()

#padd test data
for i, elem in enumerate(images_final_test):
    h, w, d = elem.shape
    new_size = max(h, w)
    img = Image.fromarray(elem, 'RGB')
    delta_h = new_size - h
    delta_w = new_size - w
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(img, padding)
    images_final_test[i] = numpy.array(new_im)

#Resize test image s
for i, elem in enumerate(images_final_test):
    size = image_size
    img = Image.fromarray(elem, 'RGB')
    images_final_test[i] = numpy.array(ImageOps.fit(img, size, Image.ANTIALIAS))

#normalize test images
for i, element in enumerate(images_final_test):
    images_final_test[i] = normalize_image(element)
#make vector from test images
final_images = make_vector(images_final_test)

#calculate accuracy
y_pred_final=clf.predict(final_images)
print("Accuracy:", metrics.accuracy_score(labels_final_test, y_pred_final))


from sklearn.metrics import classification_report
#calculate precision and recall
print(classification_report(y_pred_final,labels_final_test))

#Create confusion matrix for test data
y_test = labels_final_test
pred = y_pred_final


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, pred)
import pandas as pd
import seaborn as sns

df_cm = pd.DataFrame(conf_matrix, index = labels, columns = labels)

sns.heatmap(df_cm, cmap = "Greys", annot = True, fmt = 'g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#print names of images that identified incorrectly
names = []
images = []
for i in range(len(y_pred_final)):
    if(y_pred_final[i]!=labels_final_test[i]):
        names.append(image_for_printing_final[i][0])
        images.append(image_for_printing_final[i][1])
#print image that was classified incorrectly
plt.imshow(images[10])

