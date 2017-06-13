import logging
from collections import OrderedDict
import random
import os
import math
import numpy as np
import theano
import time

from utils.general import logMessage, logTable
from models.sectorNet import SectorNet
from models.baseNet import BaseNet

from utils.sampling import getSamplesForSubEpoch
from utils.loadData import loadSinglePatientData



def generateNetwork(configFile):

    logger = logging.getLogger(__name__)

    configInfo = {}
    execfile(configFile, configInfo)

    # The networksDict should conresponding to the import statement. 
    networksDict = {'sectorNet': SectorNet,
                    'baseNet': BaseNet}

    networkType = configInfo['networkType']

    message = 'Generating {}'.format(networkType)
    logger.info(logMessage('#', message))

    networkClass = networksDict[networkType]

    preTrainedWeights = configInfo['preTrainedWeights']
    if preTrainedWeights == '':
        message = 'We will create a new network'
        logger.info(logMessage(' ', message))
    else:
        message = 'We will use a pre trained network, '
        message += 'by using pretraintd weights, stored in {}, '.format(preTrainedWeights)
        message += 'to initialize it'
        logger.info(logMessage(' ', message))

    message = 'Begin to create the network named {}'.format(configInfo['networkName'])
    logger.info(logMessage(' ', message))
    network = networkClass(configFile)

    message = 'Sucessfully Generated {}'.format(networkType)
    logger.info(logMessage('#', message))

    return network



def trainNetwork(network, configFile):

    logger = logging.getLogger(__name__)

    message = 'Training {}'.format(network.networkType)
    logger.info(logMessage('#', message))

    configInfo = {}
    execfile(configFile, configInfo)

    trainSampleSize = configInfo['trainSampleSize']
    networkType = network.networkType
    receptiveField = network.receptiveField
    networkSummary = network.summary(trainSampleSize)

    message = 'Network Summary'
    logger.info(logMessage('*', message))
    logger.info(networkSummary)

    tableRowList = []
    tableRowList.append(['Network Type', networkType])
    tableRowList.append(['Receptive Field', receptiveField])

    logger.info(logTable(tableRowList))
    logger.info(logMessage('*', '*'))

    imageFolder = configInfo['imageFolder']
    imageGrades = configInfo['imageGrades']
    numOfPatients = configInfo['numOfPatients']
    trainValRatio = configInfo['trainValRatio']
    modals = configInfo['modals']
    useROI = configInfo['useROI']
    normType = configInfo['normType']
    trainSampleSize = configInfo['trainSampleSize']
    weightMapType = configInfo['weightMapType']


    message = 'Training Data Summary'
    logger.info(logMessage('*', message))

    tableRowList = []
    tableRowList.append(['Image Folder', imageFolder])
    tableRowList.append(['Image Grades', imageGrades])
    tableRowList.append(['Number of Patients', numOfPatients])
    tableRowList.append(['Training / validation', trainValRatio])
    tableRowList.append(['Modals', modals])
    tableRowList.append(['Use ROI', useROI])
    tableRowList.append(['Normalization Type', normType])
    tableRowList.append(['Training Samples size', trainSampleSize])
    tableRowList.append(['Weight Map Type', weightMapType])
    
    logger.info(logTable(tableRowList))
    
    logger.info(logMessage('*', '*'))

    message = 'First, read the patients data files name'
    logger.info(logMessage('*', message))

    patientsDirList = []

    for grade in imageGrades:
        gradeDir = os.path.join(imageFolder, grade)

        for patient in os.listdir(gradeDir):
            patientDir = os.path.join(gradeDir, patient)
            patientsDirList.append(patientDir)

    trainAllRatio = trainValRatio / (1.0 + trainValRatio)
    numForTrain = int(numOfPatients * trainAllRatio)
    numForVal = numOfPatients - numForTrain
    assert numForTrain > 0 and numForVal > 0

    message = 'We get {} patients files. '.format(len(patientsDirList))
    message += 'Now, we need to randomly choose '
    message += '{} for training and {} for validation'.format(numForTrain, numForVal)
    logger.info(logMessage(' ', message))

    random.shuffle(patientsDirList)

    patsDirForTrainList = patientsDirList[: numForTrain]
    patsDirForValList = patientsDirList[numForTrain: numOfPatients]

    numOfModals = len(modals)
    memoryNeededPerPatData = 0.035 * numOfModals + int(useROI) * 0.020

    numOfEpochs = configInfo['numOfEpochs']
    numOfSubEpochs = configInfo['numOfSubEpochs']
    memoryThreshold = configInfo['memoryThreshold']

    usePoolToSample = configInfo['usePoolToSample']

    assert memoryThreshold > memoryNeededPerPatData

    maxPatNumPerSubEpoch = math.floor(memoryThreshold / memoryNeededPerPatData)
    maxPatNumPerSubEpoch = int(maxPatNumPerSubEpoch)

    patDirPerSubEpochDict = {}

    if maxPatNumPerSubEpoch >= len(patsDirForTrainList):

        for subEpIdx in xrange(numOfSubEpochs):
            patDirPerSubEpochDict[subEpIdx] = patsDirForTrainList

        message = 'The memory needed for all training data is '
        message += 'less than the memory threshold, so for each '
        message += 'subepoch we can use all training data'
        logger.info(logMessage(' ', message))

    else:

        for subEpIdx in xrange(numOfSubEpochs):
            chosenPatsDir = patsDirForTrainList[:maxPatNumPerSubEpoch]
            patDirPerSubEpochDict[subEpIdx] = chosenPatsDir

            random.shuffle(patsDirForTrainList)

        message = 'The memory needed for all training data is '
        message += 'more than the memory threshold, '
        message += 'so consider the memory threshold for each'
        message += 'subepoch we can only choose {} patients data'.format(maxPatNumPerSubEpoch)
        logger.ingo(logMessage(' ', message))

    message = 'Begin Training Loops'
    logger.info(logMessage('=', message))

    numOfTrainSamplesPerSubEpoch = configInfo['numOfTrainSamplesPerSubEpoch']
    message = 'For each subepoch we will use {} '.format(numOfTrainSamplesPerSubEpoch)
    message += 'samples to train the network'
    logger.info(logMessage(' ', message))

    batchSize = configInfo['batchSize']
    message = 'The batch size is {}'.format(batchSize)
    logger.info(logMessage(' ', message))

    assert batchSize < numOfTrainSamplesPerSubEpoch

    differCreteria = lambda:{'trainLoss':[], 'trainACC':[]}
    trainSubResults = lambda:{subEpIdx: differCreteria() for subEpIdx in xrange(numOfSubEpochs)}
    trainResults = {epIdx: trainSubResults() for epIdx in xrange(numOfEpochs)}

    # For logger train results
    tableRowList = []
    tableRowList.append(['EPOCH', 'SUBEPOCH', 'Train Loss', 'Train ACC', 'Sampling Time', 'Training Time'])


    for epIdx in xrange(numOfEpochs):

        epStartTime = time.time()

        message = 'EPOCH: {}/{}'.format(epIdx + 1, numOfEpochs)
        logger.info(logMessage('+', message))

        trainEpLoss = 0
        trainEpACC = 0
        trainEpBatchNum = 0

        totalSamplingTime = 0
        totalTrainingTime = 0

        for subEpIdx in xrange(numOfSubEpochs):

            subEpStartTime = time.time()

            message = 'SUBEPOCH: {}/{}'.format(subEpIdx + 1, numOfSubEpochs)
            logger.info(logMessage('-', message))
 
            # Training
            # ####################################################################
            # Just for short statement.
            # -----------------------------------------------------------------
            message = 'Training Sampling'
            logger.info(logMessage('.', message))
            sampleStartTime = time.time()
            sampleAndLabelList = getSamplesForSubEpoch(numOfTrainSamplesPerSubEpoch,
                                                       patDirPerSubEpochDict[subEpIdx],
                                                       useROI,
                                                       modals,
                                                       normType,
                                                       trainSampleSize ,
                                                       receptiveField,
                                                       weightMapType,
                                                       usePoolToSample)

            sampleStartTime = time.time() - sampleStartTime

            shuffledSamplesList, shuffledLabelsList = sampleAndLabelList
            # -----------------------------------------------------------------
            
            batchIdxList = [batchIdx for batchIdx 
                            in xrange(0, numOfTrainSamplesPerSubEpoch, batchSize)]

            # For the last batch not to be too small.
            batchIdxList[-1] = numOfTrainSamplesPerSubEpoch

            assert len(batchIdxList) > 1

            trainSubEpLoss = 0
            trainSubEpACC = 0
            trainSubEpBatchNum = 0

            # Batch loops
            # -----------------------------------------------------------------
            message = 'Training'
            logger.info(logMessage(':', message))
            for batchIdx in batchIdxList[:-1]:

                samplesBatch = shuffledSamplesList[batchIdx:batchIdx + 1]
                samplesBatch = np.asarray(samplesBatch, dtype = theano.config.floatX)

                labelsBatch = shuffledLabelsList[batchIdx:batchIdx + 1]
                labelsBatch = np.asarray(labelsBatch, dtype = 'int32')

                trainBatchLoss, trainBatchAcc = network.trainFunc(samplesBatch, labelsBatch)

                trainSubEpLoss += trainBatchLoss
                trainSubEpACC += trainBatchAcc
                trainSubEpBatchNum += 1
            # -----------------------------------------------------------------
            subEpStartTime = time.time() - subEpStartTime - sampleStartTime

            trainEpLoss += trainSubEpLoss
            trainEpACC += trainSubEpACC
            trainEpBatchNum += trainSubEpBatchNum

            trainSubEpLoss /= trainSubEpBatchNum
            trainSubEpACC /= trainSubEpBatchNum

            trainResults[epIdx][subEpIdx]['trainLoss'].append(trainSubEpLoss)
            trainResults[epIdx][subEpIdx]['trainACC'].append(trainSubEpACC)

            totalSamplingTime += sampleStartTime
            totalTrainingTime += subEpStartTime

            subEpStartTime = '{:.3f}'.format(subEpStartTime)
            sampleStartTime = '{:.3f}'.format(sampleStartTime)
            # For logger
            # -----------------------------------------------------------------
            if subEpIdx == 0:
                tableRowList.append([epIdx + 1, 
                                     subEpIdx + 1, 
                                     trainSubEpLoss, 
                                     trainSubEpACC, 
                                     sampleStartTime, 
                                     subEpStartTime])
            else:
                tableRowList.append(['', 
                                     subEpIdx + 1, 
                                     trainSubEpLoss, 
                                     trainSubEpACC, 
                                     sampleStartTime, 
                                     subEpStartTime])

            message = 'SUBEPOCH: {}/{} '.format(subEpIdx + 1, numOfSubEpochs)
            message += 'took {} s.'.format(subEpStartTime)
            message += 'SubEpoch Train Loss: {:.6f}, '.format(trainSubEpLoss)
            message += 'Train ACC: {:.6f}'.format(trainSubEpACC)
            logger.info(logMessage('-', message))
            # ------------------------------------------------------------------

        trainEpLoss /= trainEpBatchNum
        trainEpACC /= trainEpBatchNum

        # For logger
        # -------------------------------------------------------------------
        totalSamplingTime = '{:.3}'.format(totalSamplingTime)
        totalTrainingTime = '{:.3}'.format(totalTrainingTime)

        tableRowList.append(['-', '-', '-', '-', '-', '-'])
        tableRowList.append(['', '', trainEpLoss, trainEpACC, totalSamplingTime, totalTrainingTime])
        tableRowList.append(['-', '-', '-', '-', '-', '-'])

        message = 'EPOCH: {}/{} '.format(epIdx + 1, numOfEpochs)
        message += 'took {:.3f} s.'.format(time.time() - epStartTime)
        message += 'Epoch Train Loss: {:.6f}, '.format(trainEpLoss)
        message += 'Epoch Train ACC: {:.6f}'.format(trainEpACC)
        logger.info(logMessage('+', message))
        # --------------------------------------------------------------------
        # ####################################################################

        message = 'Validation'
        logger.info(logMessage(':', message))



    message = 'The Training Results'
    logger.info(logMessage('=', message))
    logger.info(logTable(tableRowList))
    logger.info(logMessage('=', '='))


    message = 'End Training Loops'
    logger.info(logMessage('#', message))


    return trainResults



def testNetwork(network, configFile):

    logger = logging.getLogger(__name__)

    message = 'Testing {}'.format(network.networkType)
    logger.info(logMessage('#', message))

    configInfo = {}
    execfile(configFile, configInfo)

    # For summary.
    testSampleSize = configInfo['testSampleSize']
    networkType = network.networkType
    receptiveField = network.receptiveField
    networkSummary = network.summary(testSampleSize)

    message = 'Network Summary'
    logger.info(logMessage('*', message))
    logger.info(networkSummary)

    tableRowList = []
    tableRowList.append(['Network Type', networkType])
    tableRowList.append(['Receptive Field', receptiveField])

    logger.info(logTable(tableRowList))
    logger.info(logMessage('*', '*'))

    testImageFolder = configInfo['testImageFolder']
    useROITest = configInfo['useROITest']
    modals = configInfo['modals']
    normType = configInfo['normType']
    testSampleSize = configInfo['testSampleSize']

    numOfPatients = len(os.listdir(testImageFolder))


    message = 'Test Data Summary'
    logger.info(logMessage('*', message))

    tableRowList = []
    tableRowList.append(['Test Image Folder', testImageFolder])
    tableRowList.append(['Number of Patients', numOfPatients])
    tableRowList.append(['Use ROI To Test Network', useROITest])
    tableRowList.append(['Modals', modals])
    tableRowList.append(['Normalization Type in Test Process', normType])
    tableRowList.append(['Test Samples size', testSampleSize])

    logger.info(logTable(tableRowList))    
    logger.info(logMessage('*', '*'))

    message = 'First, read the patients data files name'
    logger.info(logMessage('*', message))


    message = 'We get {} patients files. '.format(len(os.listdir(testImageFolder)))
    logger.info(logMessage(' ', message))

    outputFolder = configInfo['outputFolder']

    for patient in os.listdir(testImageFolder):

        patientDir = os.path.join(testImageFolder, patient)

        segmentResultNameWithPath = os.path.join(outputFolder, patient)

        # For short statement.
        sampleWholeImageResult = sampleWholeImage(patientDir, 
                                                  useROITest, 
                                                  modals, 
                                                  normType, 
                                                  testSampleSize, 
                                                  receptiveField)

        samplesOfWholeImage = sampleWholeImageResult[0]
        labelsOfWholeImage = sampleWholeImageResult[1]
        wholeLabelCoordList = sampleWholeImageResult[2]
        imageShape = sampleWholeImageResult[3]

        segmentResult = np.zeros(imageShape, dtype = 'int32')
        segmentResultMask = np.zeros(imageShape, dtype = 'int16')

























