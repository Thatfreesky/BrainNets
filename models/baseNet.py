import numpy as np
import theano
import time
import os
import logging
import theano.tensor as T
import lasagne
from lasagne.init import HeUniform, HeNormal
from lasagne.updates import momentum, rmsprop
from lasagne.layers import (InputLayer,
                            ConcatLayer,
                            Conv3DLayer,
                            Pool3DLayer,
                            batch_norm,
                            BatchNormLayer,
                            NonlinearityLayer,
                            get_all_params, 
                            DimshuffleLayer,
                            get_output, 
                            set_all_param_values, 
                            get_all_param_values, 
                            get_output_shape, 
                            ReshapeLayer,
                            prelu)

from lasagne.regularization import regularize_network_params
from lasagne.nonlinearities import rectify, linear, softmax
from lasagne.objectives import categorical_crossentropy

from utils.general import logMessage, logTable

class BaseNet():

    def __init__(self, configFile):

        self.logger = logging.getLogger(__name__)
        self.configInfo = {}
        execfile(configFile, self.configInfo)

        assert self.configInfo['networkType'] == 'baseNet'
        self.networkType = 'baseNet'

        self.outputFolder = self.configInfo['outputFolder']

        # ----------------------------------------------------------------------------------------
        # For build BaseNet.
        self.preTrainedWeights = self.configInfo['preTrainedWeights']

        # For displaying the output shape change process through the network.
        self.modals = self.configInfo['modals']
        self.inputChannels = len(self.modals)
        self.trainSampleSize = self.configInfo['trainSampleSize']
        self.kernelShapeList = self.configInfo['kernelShapeList']
        self.kernelNumList = self.configInfo['kernelNumList']
        self.numOfClasses = self.configInfo['numOfClasses']

        assert len(self.kernelShapeList) == len(self.kernelNumList)
        assert self.numOfClasses == self.kernelNumList[-1]

        self.inputShape = (None, 
                           self.inputChannels,
                           None,
                           None,
                           None)

        self.inputVar = T.tensor5('inputVar', dtype = theano.config.floatX)
        self.targetVar = T.tensor4('targetVar', dtype = 'int32')
        self.receptiveField = 'Only after building the BaseNet, can we get the receptive filed'
        self.outputLayer = self.buildBaseNet()
        self.restoreWeights()

        # ----------------------------------------------------------------------------------------
        # For comlile train function.
        # We let the learning can be learnt
        self.weightsFolder = self.configInfo['weightsFolder']
        self.learningRate = theano.shared(np.array(self.configInfo['learningRate'], 
                                                   dtype = theano.config.floatX))
        self.learningRateDecay = np.array(self.configInfo['learningRateDecay'], 
                                                   dtype = theano.config.floatX)
        self.weightDecay = self.configInfo['weightDecay']

        self.optimizerDict = {'rmsprop':rmsprop,
                              'momentum':momentum}

        self.optimizer = self.optimizerDict[self.configInfo['optimizer']]

        self.trainFunction = self.complieTrainFunction()


        # ----------------------------------------------------------------------------------------
        # For compile val function
        self.valSampleSize = self.configInfo['valSampleSize']
        self.valFunction = self.compileValFunction()

        # ----------------------------------------------------------------------------------------
        # For compile test function
        self.testSampleSize = self.configInfo['testSampleSize']
        self.testFunction = self.compileTestFunction()



    def buildBaseNet(self, inputShape = (None, 4, 25, 25, 25), forSummary = False):

        if not forSummary:
            message = 'Building the Architecture of BaseNet'
            self.logger.info(logMessage('+', message))

        baseNet = InputLayer(self.inputShape, self.inputVar)

        if not forSummary:
            message = 'Building the convolution layers'
            self.logger.info(logMessage('-', message))

        kernelShapeListLen = len(self.kernelNumList)

        summary = '\n' + '.' * 130 + '\n'
        summary += '    {:<15} {:<50} {:<29} {:<29}\n'.format('Layer', 
                                                              'Input shape', 
                                                              'W shape', 
                                                              'Output shape')
        summary += '.' * 130 + '\n'

        summary += '{:<3} {:<15} {:<50} {:<29} {:<29}\n'.format(1, 
                                                                'Input', 
                                                                inputShape, 
                                                                '',
                                                                get_output_shape(baseNet, 
                                                                input_shapes = inputShape))

        for i in xrange(kernelShapeListLen - 1):

            kernelShape = self.kernelShapeList[i]
            kernelNum = self.kernelNumList[i]

            conv3D = Conv3DLayer(incoming = baseNet,
                                  num_filters = kernelNum,
                                  filter_size = kernelShape,
                                  W = HeNormal(gain = 'relu'),
                                  nonlinearity = linear,
                                  name = 'Conv3D{}'.format(i))

            # Just for summary the fitler shape.
            WShape = conv3D.W.get_value().shape

            summary += '{:<3} {:<15} {:<50} {:<29} {:<29}\n'.format(i + 2, 
                                                                    'Conv3D', 
                                                                    get_output_shape(baseNet, input_shapes = inputShape), 
                                                                    WShape,
                                                                    get_output_shape(conv3D, input_shapes = inputShape))

            batchNormLayer = BatchNormLayer(conv3D)
            preluLayer = prelu(batchNormLayer)
            
            concatLayerInputShape = '{:<25}{:<25}'.format(get_output_shape(conv3D, input_shapes = inputShape),
                                                           get_output_shape(baseNet, input_shapes = inputShape))

            baseNet = ConcatLayer([preluLayer, baseNet], 1, cropping = ['center', 
                                                                        'None', 
                                                                        'center', 
                                                                        'center', 
                                                                        'center'])

            summary += '    {:<15} {:<50} {:<29} {:<29}\n'.format('Concat', 
                                                                  concatLayerInputShape, 
                                                                  '',
                                                                  get_output_shape(baseNet, input_shapes = inputShape))
        if not forSummary:
            message = 'Finish Built the convolution layers'
            self.logger.info(logMessage('-', message))

            message = 'Building the last classfication layers'
            self.logger.info(logMessage('-', message))

        assert self.kernelShapeList[-1] == [1, 1, 1]

        kernelShape = self.kernelShapeList[-1]
        kernelNum = self.kernelNumList[-1]

        conv3D = Conv3DLayer(incoming = baseNet,
                              num_filters = kernelNum,
                              filter_size = kernelShape,
                              W = HeNormal(gain = 'relu'),
                              nonlinearity = linear,
                              name = 'Classfication Layer')


        receptiveFieldList = [inputShape[idx] - get_output_shape(conv3D, input_shapes = inputShape)[idx] + 1
                              for idx in xrange(-3, 0)]
        assert receptiveFieldList != []
        receptiveFieldSet = set(receptiveFieldList)
        assert len(receptiveFieldSet) == 1, (receptiveFieldSet, inputShape, get_output_shape(conv3D, input_shapes = inputShape))
        self.receptiveField = list(receptiveFieldSet)[0]

        # Just for summary the fitler shape.
        WShape = conv3D.W.get_value().shape

        summary += '{:<3} {:<15} {:<50} {:<29} {:<29}\n'.format(kernelShapeListLen + 1, 
                                                                'Conv3D', 
                                                                get_output_shape(baseNet, input_shapes = inputShape), 
                                                                WShape,
                                                                get_output_shape(conv3D, input_shapes = inputShape))

        # The output shape should be (batchSize, numOfClasses, zSize, xSize, ySize).
        # We will reshape it to (batchSize * zSize * xSize * ySize, numOfClasses),
        # because, the softmax in theano can only receive matrix.

        baseNet = DimshuffleLayer(conv3D, (0, 2, 3, 4, 1))
        summary += '    {:<15} {:<50} {:<29} {:<29}\n'.format('Dimshuffle', 
                                                              get_output_shape(conv3D, input_shapes = inputShape), 
                                                              '',
                                                              get_output_shape(baseNet, input_shapes = inputShape))

        batchSize, zSize, xSize, ySize, _ = get_output(baseNet).shape
        reshapeLayerInputShape = get_output_shape(baseNet, input_shapes = inputShape)
        baseNet = ReshapeLayer(baseNet, (batchSize * zSize * xSize * ySize, kernelNum))
        summary += '    {:<15} {:<50} {:<29} {:<29}\n'.format('Reshape', 
                                                              reshapeLayerInputShape, 
                                                              '',
                                                              get_output_shape(baseNet, input_shapes = inputShape))

        nonlinearityLayerInputShape = get_output_shape(baseNet, input_shapes = inputShape)
        baseNet = NonlinearityLayer(baseNet, softmax)
        summary += '    {:<15} {:<50} {:<29} {:<29}\n'.format('Nonlinearity', 
                                                              nonlinearityLayerInputShape, 
                                                              '',
                                                              get_output_shape(baseNet, input_shapes = inputShape))
        
        if not forSummary:
            message = 'Finish Built the last classfication layers'
            self.logger.info(logMessage('-', message))

            message = 'The Receptivr Field of BaseNet equal {}'.format(self.receptiveField)
            self.logger.info(logMessage('*', message))

            message = 'Finish Building the Architecture of BaseNet'
            self.logger.info(logMessage('+', message))

        summary += '.' * 130 + '\n'
        self._summary = summary

        return baseNet


    def complieTrainFunction(self):
        message = 'Compiling the Training Function'
        self.logger.info(logMessage('+', message))

        startTime = time.time()

        trainPrediction = get_output(self.outputLayer, 
                                     deterministic = False,
                                     batch_norm_update_averages=False, 
                                     batch_norm_use_averages=False)
        # TODO. Chack wheather the flatten style of targetvar and output are same.
        self.flattenedTargetVar = T.flatten(self.targetVar)

        trainLoss = categorical_crossentropy(trainPrediction, self.flattenedTargetVar).mean()
        weightNorm = regularize_network_params(self.outputLayer, lasagne.regularization.l2)
        trainLoss += self.weightDecay * weightNorm

        trainPredictionLabel = T.argmax(trainPrediction, axis = 1)
        trainACC = T.mean(T.eq(trainPredictionLabel, self.flattenedTargetVar), 
                          dtype = theano.config.floatX)
        
        params = get_all_params(self.outputLayer, trainable = True)
        update = self.optimizer(trainLoss, params, learning_rate = self.learningRate)

        trainFunc = theano.function([self.inputVar, self.targetVar], 
                                    [trainLoss, trainACC], 
                                    updates = update)
        
        message = 'Compiled the Training Function, spent {:.2f}s'.format(time.time()- startTime)
        self.logger.info(logMessage('+', message))

        return trainFunc



    def compileValFunction(self):

        message = 'Compiling the Validation Function'
        self.logger.info(logMessage('+', message))

        startTime = time.time()

        valPrediction = get_output(self.outputLayer, 
                                     deterministic = True,
                                     batch_norm_update_averages=False, 
                                     batch_norm_use_averages=False)
        # TODO. Chack wheather the flatten style of targetvar and output are same.
        self.flattenedTargetVar = T.flatten(self.targetVar)

        valLoss = categorical_crossentropy(valPrediction, self.flattenedTargetVar).mean()
        weightNorm = regularize_network_params(self.outputLayer, lasagne.regularization.l2)
        valLoss += self.weightDecay * weightNorm

        valPredictionLabel = T.argmax(valPrediction, axis = 1)
        valACC = T.mean(T.eq(valPredictionLabel, self.flattenedTargetVar), 
                        dtype = theano.config.floatX)

        valFunc = theano.function([self.inputVar, self.targetVar], 
                                  [valLoss, valACC])
        
        message = 'Compiled the Validation Function, spent {:.2f}s'.format(time.time()- startTime)
        self.logger.info(logMessage('+', message))

        return valFunc



    def compileTestFunction(self):

        message = 'Compiling the Test Function'
        self.logger.info(logMessage('+', message))

        startTime = time.time()

        testPrediction = get_output(self.outputLayer, 
                                    deterministic = True,
                                    batch_norm_use_averages=False)
       
        testPredictionLabel = T.argmax(testPrediction, axis = 1)

        testFunc = theano.function([self.inputVar], [testPredictionLabel])
        
        message = 'Compiled the Test Function, spent {:.2f}s'.format(time.time()- startTime)
        self.logger.info(logMessage('+', message))

        return testFunc


    def saveWeights(self, fileName):

        message = 'Save Weights in {}'.format(fileName)
        self.logger.info(logMessage('+', message))
        fileNameWithPath = os.path.join(self.weightsFolder, fileName)
        np.savez(fileNameWithPath, 
                 *get_all_param_values(self.outputLayer))



    def restoreWeights(self):

        if self.preTrainedWeights == '':
            return

        assert self.preTrainedWeights != ''

        message = 'Load Weights from {}'.format(self.preTrainedWeights)
        self.logger.info(logMessage('+', message))

        with np.load(self.preTrainedWeights) as f:
            savedWeights = [f['arr_{:d}'.format(i)] 
                            for i in range(len(f.files))]

        set_all_param_values(self.outputLayer, savedWeights)


    def summary(self, inputShape):

        if len(inputShape) == 3:
            inputShapeForSummary = (None, 
                                    len(self.modals), 
                                    inputShape[0], 
                                    inputShape[1], 
                                    inputShape[2])
        if len(inputShape) == 5:
            inputShapeForSummary = inputShape
        self.buildBaseNet(inputShapeForSummary, forSummary = True)

        return self._summary

