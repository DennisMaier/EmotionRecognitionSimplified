from tensorflow.keras.layers import  Input, Dense, Dropout, Flatten, MaxPooling2D,  Conv2D, BatchNormalization
from tensorflow.keras.models import Model

def vgg_block(layer_in, n_filters, n_conv, activation="relu",kernel_initializer="he_normal"):
	"""[summary]
	Basic VGG Building Block
	Args:
		layer_in ([type]): [description]
		n_filters ([type]): [description]
		n_conv ([type]): [description]
		activation (str, optional): [description]. Defaults to "relu".
		kernel_initializer (str, optional): [description]. Defaults to "he_normal".

	Returns:
		[type]: [description]
	"""
	chanDim = -1
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation=activation,kernel_initializer=kernel_initializer)(layer_in)
		layer_in = BatchNormalization(axis=chanDim)(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

class EmotionVGGDefault:
	"""[summary]
	Basic neural network vor 48x48x1 input images
	Returns:
		[type]: [description]
	"""
	@staticmethod
	def build(width, height, depth, classes,activation="relu",kernel_initializer="he_normal",last_activation="softmax"):
		"""[summary]
		Args:
			width ([type]): [description]
			height ([type]): [description]
			depth ([type]): [description]
			classes ([type]): [description]
			activation (str, optional): [description]. Defaults to "relu".
			kernel_initializer (str, optional): [description]. Defaults to "he_normal".
			last_activation (str, optional): [description]. Defaults to "softmax".
		Returns:
			[type]: [description]
		"""

		# define model 
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

