import logging
import random
import math
import theano
import numpy as np

import loadData
reload(loadData)
import general
reload(general)


def getSamplesForSubEpoch(numOfSamplesPerSubEpochTrain,
                          patientsDirList,
                          useROI = True,
                          modals = ['T1c', 'T1', 'T2', 'Flair'],
                          normType = 0,
                          trainSampleSize = [25, 25, 25],
                          receptiveField = 17,
                          weightMapType = 0):

    logger = logging.getLogger(__name__)

    assert len(patientsDirList) > 0

    # For more detail about weightMapType, place read the training cnofig file.
    if weightMapType == 0 or weightMapType == 1:
        foreBackRatio = 0.5
    elif weightMapType == 2:
        foreBackRatio = 0.2
    else:
        logger.error('The weightMapType should be one of (0, 1, 2), but {} given.'.format(weightMapType))
        raise ValueError

    # Only consider the square receptive field, 
    # so the size of receptive field can be represent by a number.
    # Only consider the recepticeField is odd number.
    assert receptiveField % 2 == 1

    # Only consider the trainSampleSize is large than the receptive field.
    assert all([size >= receptiveField for size in trainSampleSize])

    random.shuffle(patientsDirList)

    numOfPatients = len(patientsDirList)
    numOfSamplesPerPatient = numOfSamplesPerSubEpochTrain / numOfPatients
    remainder = numOfSamplesPerSubEpochTrain % numOfPatients

    # If we use 3 patient and need 17 samples, 
    # the numOfSamplesPerPatientList will look like [6, 6, 5]
    numOfSamplesPerPatientList = [numOfSamplesPerPatient] * numOfPatients
    [numOfSamplesPerPatientList[idx] + 1 for idx in range(remainder)]

    assert sum(numOfSamplesPerPatientList) == numOfSamplesPerSubEpochTrain

    # If foreBackRatio = 0.4, 
    # The numOfForeBackSamplesList will look like [[2, 4], [2, 4], [2, 3]]
    numOfForeBackSamplesList = [[int(num * foreBackRatio), num - int(num * foreBackRatio)] 
                                for num in numOfSamplesPerPatientList]

    # Now, the numOfForeBackSamplesList will look like [[4, 2], [2, 4], [3, 2]]
    for pair in numOfForeBackSamplesList:
        random.shuffle(pair)

    assert len(numOfForeBackSamplesList) == len(patientsDirList)
    assert all([sum(numOfForeBackSamplesList[idx]) == numOfSamplesPerPatientList[idx]
                for idx in range(numOfPatients)])

    samplesList = []
    labelsList = []

    receptiveFieldRadius = receptiveField / 2

    for idx, patientDir in enumerate(patientsDirList):

        # Also store the ROI in the patientLabelArray if useROI == True
        # patientImageArray dtype = theano.config.floatX.
        # patientLabelArray dtype = int16
        patientImageArray, patientLabelArray = loadData.loadSinglePatientData(patientDir, 
                                                                              useROI, 
                                                                              modals, 
                                                                              normType)

        assert patientImageArray.shape == (len(modals), 155, 240, 240)
        assert patientLabelArray.shape == (1 + int(useROI), 155, 240, 240), \
               '{} == {}'.format(patientLabelArray.shape, (1 + int(useROI), 155, 240, 240))

        assert isinstance(patientImageArray, np.ndarray)
        assert isinstance(patientLabelArray, np.ndarray)

        numOfForeBackSamples = numOfForeBackSamplesList[idx]
        samplesOfAPatient, labelsOfAPatient = sampleAPatient(patientImageArray, 
                                                             patientLabelArray, 
                                                             numOfForeBackSamples,
                                                             useROI,
                                                             trainSampleSize,
                                                             receptiveField,
                                                             weightMapType)
        assert isinstance(samplesOfAPatient, list)
        assert isinstance(labelsOfAPatient, list)

        # For short the assert statements.
        zSize = trainSampleSize[0]
        xSize = trainSampleSize[1]
        ySize = trainSampleSize[2]

        assert len(samplesOfAPatient) == sum(numOfForeBackSamples)
        assert all([sample.shape == (len(modals), zSize, xSize, ySize) 
                    for sample in samplesOfAPatient]), '{}'.format([s.shape 
                                                                    for s in samplesOfAPatient])

        # For short the assert statements.
        rF = receptiveField
        assert len(labelsOfAPatient) == sum(numOfForeBackSamples)
        assert all([label.shape == (zSize - rF + 1, xSize - rF + 1, ySize - rF + 1)
                    for label in labelsOfAPatient])

        # The two lists expend for each loop.
        samplesList += samplesOfAPatient
        labelsList += labelsOfAPatient

        logger.info('Get {} / {} samples from {} patients'.format(len(samplesList), 
                                                                    numOfSamplesPerSubEpochTrain, 
                                                                    idx + 1))

    # Release the memory.
    del samplesOfAPatient[:], samplesOfAPatient
    del labelsOfAPatient[:], labelsOfAPatient

    assert len(samplesList) == len(labelsList) == numOfSamplesPerSubEpochTrain

    logger.info('***************************************************')
    logger.info('Get all {} samples'.format(len(samplesList)))
    logger.info('The shape of a sample array equals: {}'.format(samplesList[0].shape))
    logger.info('The shape of a label array equals: {}'.format(labelsList[0].shape))
    logger.info('***************************************************')

    # By this way, we can keep the ralationships between the sample ant its label.
    zipSampleAndLabel = zip(samplesList, labelsList)
    random.shuffle(zipSampleAndLabel)

    shuffledSamplesList = []
    shuffledLabelsList = []

    logger.info('Shuffled the lists of samples and labels')

    # By this way, the shuffledSamplesList and shuffledLabelsList can be list directly.
    shuffledSamplesList[:], shuffledLabelsList[:] = zip(*zipSampleAndLabel)

    # Release the memory
    del samplesList[:], samplesList
    del labelsList[:], labelsList

    return shuffledSamplesList, shuffledLabelsList


def sampleAPatient(patientImageArray, 
                   patientLabelArray, 
                   numOfForeBackSamples,
                   useROI,
                   trainSampleSize,
                   receptiveField,
                   weightMapType):

    logger = logging.getLogger(__name__)

    gTArray = patientLabelArray[0]

    foreMask, backMask = getForeAndBackMask(patientLabelArray, 
                                            useROI,
                                            weightMapType)

    numOfForeSamples, numOfBackSamples = numOfForeBackSamples
    # Foreground samples coordinate range.
    logger.debug('Get samples coordinates by foreMask.')
    foreSamplesCdList = getSamplesCoordinateList(foreMask, 
                                                 numOfForeSamples, 
                                                 trainSampleSize)
    logger.debug('Get samples coordinates by backMask.')
    backSamplesCdList = getSamplesCoordinateList(backMask, 
                                                 numOfBackSamples, 
                                                 trainSampleSize)

    samplesCdList = foreSamplesCdList + backSamplesCdList

    samplesOfAPatient = getSamplesOfAPatient(patientImageArray, 
                                             samplesCdList,
                                             trainSampleSize)
    labelsOfAPatient = getLabelsOfAPatient(gTArray, 
                                           samplesCdList, 
                                           receptiveField, 
                                           trainSampleSize)

    return samplesOfAPatient, labelsOfAPatient


def getSamplesOfAPatient(patientImageArray, samplesCdList, trainSampleSize):

    logger = logging.getLogger(__name__)

    samplesOfAPatient = []

    for sampleCd in samplesCdList:

        # sampleCd is a list.
        sampleArrayList = []

        for modal in patientImageArray:
            zLeft = sampleCd[0][0]
            zRight = sampleCd[0][1]
            xLeft = sampleCd[1][0]
            xRight = sampleCd[1][1]
            yLeft = sampleCd[2][0]
            yRight = sampleCd[2][1]
            sampleArrayList.append(modal[zLeft:zRight, xLeft: xRight, yLeft: yRight])

            assert sampleArrayList[-1].shape == tuple(trainSampleSize), \
                   '{}:{}:{}'.format(sampleArrayList[-1].shape, 
                                     tuple(trainSampleSize), 
                                     sampleCd)

        sampleArray = np.asarray(sampleArrayList, dtype = theano.config.floatX)
        samplesOfAPatient.append(sampleArray)

    return samplesOfAPatient



def getLabelsOfAPatient(gTArray, samplesCdList, receptiveField, trainSampleSize):

    logger = logging.getLogger(__name__)

    labelsOfAPatient = []
    labelSize = [axleSize - receptiveField + 1 
                 for axleSize in trainSampleSize]
    labelSize = tuple(labelSize)

    rFRadius = receptiveField / 2

    for sampleCd in samplesCdList:
        zLeft = sampleCd[0][0] + rFRadius
        zRight = sampleCd[0][1] - rFRadius
        xLeft = sampleCd[1][0] + rFRadius
        xRight = sampleCd[1][1] - rFRadius
        yLeft = sampleCd[2][0] + rFRadius
        yRight = sampleCd[2][1] - rFRadius

        labelsOfAPatient.append(gTArray[zLeft:zRight, xLeft: xRight, yLeft: yRight])
        assert labelsOfAPatient[-1].shape == labelSize, \
               '{}:{}'.format(labelsOfAPatient[-1].shape, labelSize)

    return labelsOfAPatient



def getSamplesCoordinateList(mask, numOfSamples, trainSampleSize):

    logger = logging.getLogger(__name__)

    maskShape = mask.shape
    # If trainSampleSize = [25, 25, 25],
    # the centerLocOfSampleSize = [13, 13, 13]
    centerLocOfSampleSize = [int(math.ceil(axle / 2))
                             for axle in trainSampleSize]

    logger.debug('centerLocOfSampleSize: {}'.format(centerLocOfSampleSize))
    # In fact, I am not sure the axles is this order (z, x, y).
    # But no matter at here.
    halfLZAxle = centerLocOfSampleSize[0] - 1
    halfRZAxle = trainSampleSize[0] - centerLocOfSampleSize[0]
    halfLXAxle = centerLocOfSampleSize[1] - 1
    halfRXAxle = trainSampleSize[1] - centerLocOfSampleSize[1]
    halfLYAxle = centerLocOfSampleSize[2] - 1
    halfRYAxle = trainSampleSize[2] - centerLocOfSampleSize[2]

    # The halfAxles = [[12, 12],[12, 12], [12, 12]]
    halfAxles = [[halfLZAxle, halfRZAxle], 
                 [halfLXAxle, halfRXAxle], 
                 [halfLYAxle, halfRYAxle]]

    assert [sum(pair) + 1 for pair in halfAxles] == trainSampleSize, \
            'halfAxles: {}, trainSampleSize: {}'.format(halfAxles, trainSampleSize)

    # TODO. The effluence of the dtype of those ROI array.
    centerLocROI = np.zeros(maskShape, dtype = 'int16')
    # Attention, the index of python is begin from 0
    centerLocROI[halfLZAxle:maskShape[0] - halfRZAxle, 
                 halfLXAxle:maskShape[1] - halfRXAxle,
                 halfLYAxle:maskShape[2] - halfRYAxle] = 1

    # Just for assert statement.
    checkArray = centerLocROI[halfLZAxle:maskShape[0] - halfRZAxle, 
                              halfLXAxle:maskShape[1] - halfRXAxle,
                              halfLYAxle:maskShape[2] - halfRYAxle]
    assert all([checkArray.shape[idx] + trainSampleSize[idx] - 1 == maskShape[idx]
                for idx in range(len(trainSampleSize))])
    del checkArray

    ROIToApply = mask * centerLocROI

    # Make the element in ROIToApply to be a probality for the np.random.choice function.
    ROIToApply = ROIToApply / (1.0 * np.sum(ROIToApply))

    # To be a vector
    ROIToApplyFlattened = ROIToApply.flatten()
    
    centerVoxelIndexList = np.random.choice(ROIToApply.size,
                                            size = numOfSamples,
                                            replace = True,
                                            p = ROIToApplyFlattened)

    centerVoxelsCoord = np.asarray(np.unravel_index(centerVoxelIndexList, maskShape), dtype = 'int16')

    # Python may auto del those reference after the function finish.
    # So, this step may no use.
    del ROIToApplyFlattened, centerVoxelIndexList

    # From center voxels coordinate to sample coordinate.
    halfAxles = np.asarray(halfAxles, dtype = 'int16')
    samplesLCoord = centerVoxelsCoord - halfAxles[:, np.newaxis, 0]
    samplesRCoord = centerVoxelsCoord + halfAxles[:, np.newaxis, 1] + 1

    # Only for assert.
    trainSampleSizeArray = np.asarray(trainSampleSize, dtype = 'int16')
    trainSampleSizeArray.reshape((3, 1))

    assert np.all(samplesRCoord - samplesLCoord == trainSampleSizeArray[:, np.newaxis])

    samplesCoord = []

    for idx in range(numOfSamples):

        # I think here we should better to convert the array type to int.
        lCoord = (samplesLCoord[0][idx], samplesLCoord[1][idx], samplesLCoord[2][idx])
        rCoord = (samplesRCoord[0][idx], samplesRCoord[1][idx], samplesRCoord[2][idx])

        # For assert
        rangeList = [rCoord[idx] - lCoord[idx] for idx in xrange(len(lCoord))]
        assert rangeList == trainSampleSize, '{}:{}'.format(rangeList, trainSampleSize)

        # sampleCoord is a list.
        sampleCoord = zip(lCoord, rCoord)
        samplesCoord.append(sampleCoord)

    return samplesCoord

    # TODO. A lot of thinks about np array type need to check.



def getForeAndBackMask(patientLabelArray,
                       useROI,
                       weightMapType):

    gTArray = patientLabelArray[0]
    if useROI:
        # For brain ragion.
        ROIArray = patientLabelArray[1]
    else:
        # For all image region.
        ROIArray = np.ones(gTArray.shape, dtype = 'int16')

    assert ROIArray.shape == gTArray.shape

    # For tumor region.
    foreMask = (gTArray > 0).astype('int16')
    # For normal brain region.
    backMask = (ROIArray > 0) * (gTArray == 0)
    backMask = backMask.astype('int16')
    # TODO. For baskMask, we may also need to assign different weights
    # to different region.

    if weightMapType != 0:

        countGTArray = general.numpyArrayCounter(gTArray)
        # This dict looks like {0: 8780214, 1: 49624, 2: 23689, 3: 2113, 4: 72360}
        countGTArrayDict = dict(countGTArray)
        # Now, the dict looks like {1: 49624, 2: 23689, 3: 2113, 4: 72360}
        del countGTArrayDict[0]

        allNumber = float(sum(countGTArrayDict.values()))
        # Now, the dict looks like {1: 3, 2: 7, 3: 70, 4: 3}
        for key in countGTArrayDict.keys():
            countGTArrayDict[key] = countGTArrayDict[key] / allNumber
            countGTArrayDict[key] = math.ceil(countGTArrayDict[key])
            countGTArrayDict[key] = int(countGTArrayDict[key])

        # We need to create a weighted foreMask.
        # First, we clean the plain foreMask.
        foreMask = foreMask * 0
        # Than, for each substructure, assign the weight to its mask value.
        for key in countGTArrayDict.keys():

            tempArray = (gTArray == key).astype(int)
            tempArray *= countGTArrayDict[key]
            foreMask += tempArray

        # tempArray is an array, so can not use del tempArray.
        del tempArray
        foreMask = foreMask.astype('int16')

        assert general.numpyArrayCounter(foreMask) == countGTArray

    return foreMask, backMask







