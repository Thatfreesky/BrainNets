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
                            DropoutLayer, 
                            prelu)

from lasagne.regularization import regularize_network_params
from lasagne.nonlinearities import rectify, linear, softmax
from lasagne.objectives import categorical_crossentropy

from utils.general import logMessage, logTable
from myLayers import DilatedConv3DLayer

class Dilated3DNet():

    def __init__(self, configFile):

        self.logger = logging.getLogger(__name__)
        self.configInfo = {}
        execfile(configFile, self.configInfo)

        assert self.configInfo['networkType'] == 'dilated3DNet'
        self.networkType = 'dilated3DNet'

        self.outputFolder = self.configInfo['outputFolder']

        # ----------------------------------------------------------------------------------------
        # For build Dilated3DNet.
        self.preTrainedWeights = self.configInfo['preTrainedWeights']

        # For displaying the output shape change process through the network.
        self.modals = self.configInfo['modals']
        self.inputChannels = len(self.modals)
        self.trainSampleSize = self.configInfo['trainSampleSize']
        self.kernelShapeList = self.configInfo['kernelShapeList']
        self.kernelNumList = self.configInfo['kernelNumList']
        self.dilatedFactorList = self.configInfo['dilatedFactorList']
        self.numOfClasses = self.configInfo['numOfClasses']
        self.dropoutRates = self.configInfo['dropoutRates']

        assert len(self.kernelShapeList) == len(self.kernelNumList)
        assert len(self.kernelShapeList) == len(self.dilatedFactorList)
        assert self.numOfClasses == self.kernelNumList[-1]

        self.inputShape = (None, 
                           self.inputChannels,
                           None,
                           None,
                           None)

        self.inputVar = T.tensor5('inputVar', dtype = theano.config.floatX)
        self.targetVar = T.tensor4('targetVar', dtype = 'int32')
        self.receptiveField = 'Only after building the dilated3DNet, can we get the receptive filed'
        self.outputLayer = self.buildDilated3DNet()
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



    def buildDilated3DNet(self, inputShape = (None, 4, 25, 25, 25)):

        summaryRowList = [['-', '-', '-', '-', '-', '-']]
        summaryRowList.append(['Numbering', 'Layer', 'Input Shape', '', 'W Shape', 'Output Shape'])
        summaryRowList.append(['-', '-', '-', '-', '-', '-'])
        dilated3DNet = InputLayer(self.inputShape, self.inputVar, name = 'InputLayer')
        # ........................................................................................
        # For summary
        num = 1
        layerName = 'Input'
        inputS1 = inputShape
        inputS2 = ''
        WShape = ''
        outputS = get_output_shape(dilated3DNet, input_shapes = inputShape)
        summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
        # ........................................................................................
        layerBlockNum = len(self.kernelNumList) - 1

        for idx in xrange(layerBlockNum):

            dilatedLayer = DilatedConv3DLayer(dilated3DNet, 
                                              self.kernelNumList[idx], 
                                              self.kernelShapeList[idx], 
                                              self.dilatedFactorList[idx], 
                                              W = HeNormal(gain = 'relu'),
                                              nonlinearity = linear)
            # ....................................................................................
            # For summary
            num = idx + 2
            layerName = 'Dilated'
            inputS1 = get_output_shape(dilated3DNet, input_shapes = inputShape)
            inputS2 = ''
            WShape = dilatedLayer.W.get_value().shape
            outputS = get_output_shape(dilatedLayer, input_shapes = inputShape)
            summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
            # ....................................................................................

            batchNormLayer = BatchNormLayer(dilatedLayer)
            preluLayer = prelu(batchNormLayer)
            concatLayer = ConcatLayer([preluLayer, dilatedLayer], 1, cropping = ['center', 
                                                                                  'None', 
                                                                                  'center', 
                                                                                  'center', 
                                                                                  'center'])
            # ....................................................................................
            # For summary
            num = ''
            layerName = 'Concat'
            inputS1 = get_output_shape(dilatedLayer, input_shapes = inputShape)
            inputS2 = get_output_shape(dilated3DNet, input_shapes = inputShape)
            WShape = ''
            outputS = get_output_shape(concatLayer, input_shapes = inputShape)
            summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
            # ....................................................................................

            dilated3DNet = DropoutLayer(concatLayer, self.dropoutRates)


        dilatedLayer = DilatedConv3DLayer(dilated3DNet, 
                                          self.kernelNumList[-1], 
                                          self.kernelShapeList[-1], 
                                          self.dilatedFactorList[-1], 
                                          W = HeNormal(gain = 'relu'),
                                          nonlinearity = linear)
        # ....................................................................................
        # For summary
        num = layerBlockNum + 1
        layerName = 'Dilated'
        inputS1 = get_output_shape(dilated3DNet, input_shapes = inputShape)
        inputS2 = ''
        WShape = dilatedLayer.W.get_value().shape
        outputS = get_output_shape(dilatedLayer, input_shapes = inputShape)
        summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
        # ....................................................................................

        # For receptive field
        receptiveFieldArray = np.asarray(inputShape)[2:] - np.asarray(outputS)[2:] + 1
        assert not np.any(receptiveFieldArray - np.mean(receptiveFieldArray))
        self.receptiveField = int(np.mean(receptiveFieldArray))

        dimshuffleLayer = DimshuffleLayer(dilatedLayer, (0, 2, 3, 4, 1))
        # ....................................................................................
        # For summary
        num = ''
        layerName = 'Dimshuffle'
        inputS1 = get_output_shape(dilatedLayer, input_shapes = inputShape)
        inputS2 = ''
        WShape = ''
        outputS = get_output_shape(dimshuffleLayer, input_shapes = inputShape)
        summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
        # ....................................................................................

        batchSize, zSize, xSize, ySize, kernelNum = get_output(dimshuffleLayer).shape
        print get_output(dimshuffleLayer).shape, kernelNum
        reshapeLayer = ReshapeLayer(dimshuffleLayer, (batchSize * zSize * xSize * ySize, kernelNum))
        # ....................................................................................
        # For summary
        num = ''
        layerName = 'Reshape'
        inputS1 = get_output_shape(dimshuffleLayer, input_shapes = inputShape)
        inputS2 = ''
        WShape = ''
        outputS = get_output_shape(reshapeLayer, input_shapes = inputShape)
        summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
        # ....................................................................................

        dilated3DNet = NonlinearityLayer(reshapeLayer, softmax)
        # ....................................................................................
        # For summary
        num = ''
        layerName = 'Nonlinearity'
        inputS1 = get_output_shape(reshapeLayer, input_shapes = inputShape)
        inputS2 = ''
        WShape = ''
        outputS = get_output_shape(dilated3DNet, input_shapes = inputShape)
        summaryRowList.append([num, layerName, inputS1, inputS2, WShape, outputS])
        summaryRowList.append(['-', '-', '-', '-', '-', '-'])
        # ....................................................................................
        self._summary = summaryRowList

        return dilated3DNet

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
        self.buildDilated3DNet(inputShapeForSummary)

        return self._summary

