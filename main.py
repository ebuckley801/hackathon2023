#install numpy
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tensorflow.python.keras import layers, models
from keras import datasets

# scale data down so all values are 0 - 1
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

#define class names list and visualize 16 images from the dataset

class_names = ['Plane', 'Car', 'Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
    plt.subplot(4,4,i+1) #4x4 grid w each iteration choosing one of these places in grid to place next image
    plt.xticks([]) #
    plt.yticks([]) # no coordinates that annoy us
    plt.imshow(training_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]]) # getting label of particular image, the number and passing that number as the index for the class list. if image level is 3, == cat

    plt.show()


#building and training the model
#reducing amount of images in the neural network
#could train on first 5000 for training images and labels
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


#training the model
'''


#watch neural networks simply explained video
#adding convolutional layers
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(32,32,3))) # horse has long legs, cat has pointy ears, plane has wings, etc
model.add(layers.MaxPooling2D((2,2))) #reduces layer to essential information
model.add(layers.Conv2D(64,(3,3), activation = 'relu')) # process results
model.add(layers.MaxPooling2D(2,2)) # reduces again
model.add(layers.Conv2D(64,(3,3),activation='relu')) #scans again

#flatten now
# ex: 10x10 matrix turned in ot straight layer of 100
model.add(layers.Flatten())

#put one more dense layer of complexity in between which is the output layer
model.add(layers.Dense(64,activation='relu'))

#scaling so we get proababilty for each classification
#softmax scales all results so they add up to 1. dist of probability so one particulat answer is the case
model.add(layers.Dense(10,activation='softmax'))

#16:25
#compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#epochs 10 so model sees the data 10 times
model.fit(training_images,training_labels, epochs=10,validation_data=(testing_images,testing_labels))


#training the neural network
# test,evaluate,save so we dont need to train it every time we run the script

#loss is numerical value showing how off it is
# how much percent of testing example were classified correctly
loss,accuracy = model.evaluate(testing_images,testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')

#model = models.load_model()
'''
#now going to take images from internet and classify them
model = models.load_model('image_classifier.model')

#using pixabay.com
img = cv.imread('horse.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')