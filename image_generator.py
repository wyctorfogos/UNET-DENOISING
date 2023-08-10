import os
import random
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2


class UNet():
	def __init__(self):
		self.input_dimension=(32,32,3)
		
	def double_conv_block(self,x, n_filters):
		# Conv2D then ReLU activation
		x = layers.Conv2D(n_filters, 3, strides=(1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
		# Conv2D then ReLU activation
		x = layers.Conv2D(n_filters, 3, strides=(1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
		return x
	
	def downsample_block(self,x, n_filters):
		f = self.double_conv_block(x, n_filters)
		p = layers.MaxPool2D(2)(f)
		p = layers.Dropout(0.3)(p)
		return f, p 
	
	def upsample_block(self,x, conv_features, n_filters):
		# upsample
		x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
		# concatenate
		x = layers.concatenate([x, conv_features])
		# dropout
		x = layers.Dropout(0.3)(x)
		# Conv2D twice with ReLU activation
		x = self.double_conv_block(x, n_filters)
		
		return x
	
	def build_model(self):
		inputs = layers.Input(shape=self.input_dimension)
		# 1 - downsample
		f1, p1 = self.downsample_block(inputs, 64)
	    # 2 - downsample
		f2, p2 = self.downsample_block(p1, 128)
	    # 3 - downsample
		f3, p3 = self.downsample_block(p2, 256)
	    # 4 - downsample
		f4, p4 = self.downsample_block(p3, 512)
		# 5 - bottleneck
		bottleneck = self.double_conv_block(p4, 1024)
		# decoder: expanding path - upsample
		# 6 - upsample
		u6 = self.upsample_block(bottleneck, f4, 512)
	    # 7 - upsample
		u7 = self.upsample_block(u6, f3, 256)
	    # 8 - upsample
		u8 = self.upsample_block(u7, f2, 128)
	    # 9 - upsample
		u9 = self.upsample_block(u8, f1, 64)
	    # outputs
		outputs = layers.Conv2D(3, 1, padding="same", activation = "sigmoid")(u9)
        # unet model with Keras Functional API
		unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
		return unet_model

def plot_image_samples():
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i])
	plt.show()

def plot_image_samples_with_noise():
	plt.title("Image with noise")
	plt.subplot(2,1,1)
	plt.imshow(train_images[0])
	plt.subplot(2,1,2)
	plt.imshow(train_image_with_noise[0])
	plt.show()

def plot_prediction(predct,epoch):
	#print(predct[0].shape)
	plt.title("Predicted image")
	plt.title("Image with noise")
	plt.subplot(1,4,1)
	plt.imshow(train_images[0])
	plt.subplot(1,4,2)
	plt.imshow(train_image_with_noise[0])
	plt.subplot(1,4,3)
	plt.imshow(predct[0])
	plt.subplot(1,4,4)
	plt.imshow(abs(train_image_with_noise[0]-predct[0]))
	plt.savefig("Epoch: {}".format(epoch))
	plt.show()


def load_dataset():
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

	# Normalize pixel values to be between 0 and 1
	train_images, test_images = train_images / 255.0, test_images / 255.0
	return train_images,test_images
##Add noise
uni_noise=np.zeros((32,32,3),dtype=np.uint8)
cv2.randu(uni_noise,0,255)
uni_noise=(uni_noise).astype(np.uint8)/255.0

train_images,test_images=load_dataset()

train_image_with_noise=train_images +uni_noise*0.5
test_image_with_noise=test_images +uni_noise*0.5


plot_image_samples_with_noise()

##Load model UNet
Unet_model= UNet().build_model()
#Unet.summary()
Unet_model.compile(optimizer='adam', loss= tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])
#Before the trainning process
#predct=Unet_model.predict(np.expand_dims(train_images[0], axis=0))
#plot_prediction(predct)

#model_history = Unet_model.fit(train_image_with_noise, train_images, epochs=1, batch_size=16, validation_data=(test_image_with_noise, test_images))
batch_size = 258
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for i in range(0, len(train_images), batch_size):
        x_batch = train_image_with_noise[i:i+batch_size]
        y_batch = train_images[i:i+batch_size]
        
        loss, accuracy = Unet_model.train_on_batch(x_batch, y_batch)
        print(f"Batch {i//batch_size + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
    predicted_img = Unet_model.predict(np.expand_dims(train_images[0], axis=0))
    plot_prediction(predicted_img,epoch)

Unet_model.save('Unet_model.h5')

#After the trainning process
#predct=Unet_model.predict(np.expand_dims(train_images[0], axis=0))
#plot_prediction(predct)