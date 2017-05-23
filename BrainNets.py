import numpy as np
import theano
import logging
import theano.tensor as T
from lasagne.init import HeUniform
from lasagne.updates import adam, momentum
from lasagne.layers import (InputLayer, ConcatLayer, Conv3DLayer, Pool3DLayer, 
                            NonlinearityLayer, get_all_params, DimshuffleLayer,
                            get_output, set_all_param_values, get_output_shape, get_all_layers, ReshapeLayer)
from lasagne.nonlinearities import rectify, linear, softmax

from lasagne.objectives import categorical_crossentropy, binary_crossentropy

class Network():
    def __init__(self,
                 input_shape=(None, 1, 5, 5, 5)):
        self.cubeSize = input_shape[-1]

        # Theano variables
        self.input_var = T.tensor5('input_var')  # input image
        self.target_var = T.ivector('target_var')  # target

        self.logger = logging.getLogger(__name__)
        
        input_layer = InputLayer(input_shape, self.input_var)
        self.logger.info('The shape of input layer is {}'.format(get_output_shape(input_layer)))

        hidden_layer1 = Conv3DLayer(incoming = input_layer, num_filters = 16, filter_size = (3, 3, 3), W=HeUniform(gain='relu'), nonlinearity = rectify)
        self.logger.info('The shape of first hidden layer is {}'.format(get_output_shape(hidden_layer1)))

        hidden_layer2 = Conv3DLayer(incoming = hidden_layer1, num_filters = 32, filter_size = (3, 3, 3), W=HeUniform(gain='relu'), nonlinearity = rectify)
        self.logger.info('The shape of second hidden layer is {}'.format(get_output_shape(hidden_layer2)))

        hidden_layer3 = Conv3DLayer(incoming = hidden_layer2, num_filters = 2, filter_size = (1, 1, 1), W=HeUniform(gain='relu'), nonlinearity = rectify)
        self.logger.info('The shape of third hidden layer is {}'.format(get_output_shape(hidden_layer3)))

        shuffledLayer = DimshuffleLayer(hidden_layer3, (0, 2, 3, 4, 1))
        self.logger.info('The shape of shuffled layer is {}'.format(get_output_shape(shuffledLayer)))

        reshapedLayer = ReshapeLayer(shuffledLayer, ([0], -1))
        self.logger.info('The shape of reshaped layer is {}'.format(get_output_shape(reshapedLayer)))

        self.output_layer = NonlinearityLayer(reshapedLayer, softmax)
        self.logger.info('The shape of output layer is {}'.format(get_output_shape(self.output_layer)))


    def trainFunction(self):

        trainPrediction = get_output(self.output_layer)
        trainLoss = categorical_crossentropy(trainPrediction, self.target_var).mean()
        # zeroLoss = categorical_crossentropy(self.target_var, self.target_var)
        trainPairLoss = categorical_crossentropy(trainPrediction, self.target_var)

        trainACC = T.mean(T.eq(T.argmax(trainPrediction, axis = 1), self.target_var), dtype = theano.config.floatX)

        trainLocation = T.argmax(trainPrediction, axis = 1)
        params = get_all_params(self.output_layer, trainable = True)
        update = momentum(trainLoss, params, learning_rate = 0.001, momentum=0.9)
        trainFunc = theano.function([self.input_var, self.target_var], [self.target_var, trainPrediction, trainACC, trainLocation, trainLoss, trainPairLoss], updates = update)
        
        return trainFunc

    def valAndTestFunction(self):

        valAndTestPrediction = get_output(self.output_layer, deterministic = True)
        valAndTestLoss = categorical_crossentropy(valAndTestPrediction, self.target_var).mean()
        
        valAndTestACC = T.mean(T.eq(T.argmax(valAndTestPrediction, axis = 1), self.target_var), dtype = theano.config.floatX)

        location = T.argmax(valAndTestPrediction, axis = 1)
        # self.logger.debug('The T.argmax(valAndTestPrediction, axis=1) equals {} and the target_var equals {}'.format(T.argmax(valAndTestPrediction, axis = 1), self.target_var))

        valAndTestFunc = theano.function([self.input_var, self.target_var], [valAndTestLoss, valAndTestACC, location, self.target_var])

        return valAndTestFunc