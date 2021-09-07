import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Dense, Dropout, Flatten,AveragePooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.models import Sequential,Model
from keras.layers.merge import concatenate
from keras.layers import add

def vgg_block(layer_in, n_filters, n_conv, activation="relu",kernel_initializer="he_normal"):
	chanDim = -1
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation=activation,kernel_initializer=kernel_initializer)(layer_in)
		layer_in = BatchNormalization(axis=chanDim)(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

class EmotionVGGDefault:
	@staticmethod
	def build(width, height, depth, classes,activation="relu",kernel_initializer="he_normal",last_activation="softmax"):
		# define model input
		visible = Input(shape=(height, width, depth))
		layer = vgg_block(visible, 32, 2,activation=activation,kernel_initializer=kernel_initializer)
		layer = vgg_block(layer, 64, 2,activation=activation,kernel_initializer=kernel_initializer)
		layer = vgg_block(layer, 128, 4,activation=activation,kernel_initializer=kernel_initializer)

		x =  Flatten()(layer)
		x =  Dense(64, kernel_initializer=kernel_initializer,activation=activation)(x)
		x =  BatchNormalization()(x)
		x =  Dropout(0.5)(x)
		x =  Dense(64, kernel_initializer=kernel_initializer,activation=activation)(x)
		x =  BatchNormalization()(x)
		x =  Dropout(0.5)(x)

		sigmoid_output = Dense(classes,name="output", activation=last_activation)(x)
		model = Model(visible,sigmoid_output)
		name ="EmotionVGGDefault"
		return model, name

