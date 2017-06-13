import numpy as np
import theano
import time
import logging
import theano.tensor as T
from lasagne.init import HeUniform
from lasagne.updates import adam, momentum
from lasagne.layers import (InputLayer, ConcatLayer, Conv3DLayer, Pool3DLayer, batch_norm,
                            NonlinearityLayer, get_all_params, DimshuffleLayer,
                            get_output, set_all_param_values, get_output_shape, get_all_layers, ReshapeLayer,get_all_param_values)
from lasagne.nonlinearities import rectify, linear, softmax

from lasagne.objectives import categorical_crossentropy, binary_crossentropy

class SectorNet():

    def __init__(self, modelConfigFile):

        self.logger = logging.getLogger(__name__)

        self.modelConfig = {}
        execfile(modelConfigFile, self.modelConfig)

        self.numOfFMs = self.modelConfig['numOfFMs']
        self.layerCategory = self.modelConfig['layerCategory']
        self.numOfOutputClass = self.modelConfig['numOfOutputClass']
        self.numOfInputChannels = self.modelConfig['numOfInputChannels']


        self.receptiveField = reduce(lambda x, y: x + y, 
                                     [int(layer[-1]) - 1 for layer in self.layerCategory], 
                                     1)
        
        self.inputShape = (None, 
                           self.numOfInputChannels, 
                           self.receptiveField,
                           self.receptiveField,
                           self.receptiveField)

        self.inputVar = T.tensor5('inputVar', dtype = theano.config.floatX)
        self.targetVar = T.ivector('targetVar')

        self.sectorNet = self.buildSectorNet()



    def buildSectorNet(self):

        sectorNet = InputLayer(self.inputShape, self.inputVar)

        for i, layer in enumerate(self.layerCategory):

            self.logger.debug('Build {}th conv layer'.format(i))
            self.logger.debug('The output shape of {}th layer equal {}'.format(i - 1, get_output_shape(sectorNet)))

            kernelXDim = int(layer[-1])
            kernelDim = (kernelXDim,) * 3


            conv3D = batch_norm(Conv3DLayer(incoming = sectorNet, 
                                            num_filters = self.numOfFMs[i], 
                                            filter_size = kernelDim, 
                                            W = HeUniform(gain = 'relu'), 
                                            nonlinearity = rectify,
                                            name = 'Conv3D'))
            self.logger.debug('The shape of {}th conv3D layer equals {}'.format(i, get_output_shape(conv3D)))

            sectorNet = ConcatLayer([conv3D, sectorNet], 1, cropping = ['center', 'None', 'center', 'center', 'center'])

            self.logger.debug('The shape of {}th concat layer equals {}'.format(i, get_output_shape(sectorNet)))

        assert get_output_shape(sectorNet) == (None, sum(self.numOfFMs) + 1, 1, 1, 1)

        sectorNet = batch_norm(Conv3DLayer(incoming = sectorNet,
                                           num_filters = 2,
                                           filter_size = (1, 1, 1),
                                           W = HeUniform(gain = 'relu')))

        self.logger.debug('The shape of last con3D layer equals {}'.format(get_output_shape(sectorNet)))

        sectorNet = ReshapeLayer(sectorNet, ([0], -1))
        self.logger.debug('The shape of ReshapeLayer equals {}'.format(get_output_shape(sectorNet)))

        sectorNet = NonlinearityLayer(sectorNet, softmax)
        self.logger.debug('The shape of output layer, i.e. NonlinearityLayer, equals {}'.format(get_output_shape(sectorNet)))

        assert get_output_shape(sectorNet) == (None, self.numOfOutputClass)

        return sectorNet


    def trainFunction(self):

        startTime = time.time()

        trainPrediction = get_output(self.sectorNet)
        trainLoss = categorical_crossentropy(trainPrediction, self.targetVar).mean()
        trainACC = T.mean(T.eq(T.argmax(trainPrediction, axis = 1), self.targetVar), 
                               dtype = theano.config.floatX)
  
        params = get_all_params(self.sectorNet, trainable = True)
        update = momentum(trainLoss, params, learning_rate = 0.001, momentum=0.9)
        trainFunc = theano.function([self.inputVar, self.targetVar], [trainLoss, trainACC], updates = update)
        self.logger.info('Compiling the train function, which spends {}.'.format(time.time() - startTime))
        
        return trainFunc

    def valAndTestFunction(self):

        startTime = time.time()

        valAndTestPrediction = get_output(self.sectorNet, deterministic = True)
        valAndTestLoss = categorical_crossentropy(valAndTestPrediction, self.targetVar).mean()
        
        valAndTestACC = T.mean(T.eq(T.argmax(valAndTestPrediction, axis = 1), self.targetVar), dtype = theano.config.floatX)
        valAndTestFunc = theano.function([self.inputVar, self.targetVar], [valAndTestLoss, valAndTestACC])
        self.logger.info('Compiling the val/test function, which spends {}.'.format(time.time() - startTime))

        return valAndTestFunc 



    def summary(self, light=False):
        """ Print a summary of the network architecture """

        layer_list = get_all_layers(self.sectorNet)

        def filter_function(layer):
            """ We only display the layers in the list below"""
            return np.any([isinstance(layer, layer_type) for layer_type in
                           [InputLayer, Conv3DLayer, Pool3DLayer, ConcatLayer]])

        layer_list = filter(filter_function, layer_list)
        output_shape_list = map(get_output_shape, layer_list)
        layer_name_function = lambda s: str(s).split('.')[3].split('Layer')[0]

        if not light:
            print('-' * 75)
            print 'Warning : all the layers are not displayed \n'
            print '    {:<15} {:<20} {:<20}'.format('Layer', 'Output shape', 'W shape')

            for i, (layer, output_shape) in enumerate(zip(layer_list, output_shape_list)):
                if hasattr(layer, 'W'):
                    input_shape = layer.W.get_value().shape
                else:
                    input_shape = ''

                print '{:<3} {:<15} {:<20} {:<20}'.format(i + 1, layer_name_function(layer), output_shape, input_shape)
                if isinstance(layer, Pool3DLayer):
                    print('')

        print '\nNumber of Convolutional layers : {}'.format(
            len(filter(lambda x: isinstance(x, Conv3DLayer), layer_list)))
        print 'Number of parameters : {}'.format(np.sum(map(np.size, get_all_param_values(self.sectorNet))))
        print('-' * 75)




        



