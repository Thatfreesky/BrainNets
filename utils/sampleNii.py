import logging
import random
import math
import theano
import numpy as np
import multiprocessing as mp

import loadNiiData
reload(loadNiiData)
import general
reload(general)
from general import logMessage, logTable


def getSamplesForSubEpoch(numOfSamplesPerSubEpochTrain,
                          patientsDirList,
                          useROI = True,
                          modals = ['t1ce', 't1', 't2', 'flair'],
                          normType = 'normImage',
                          trainSampleSize = [25, 25, 25],
                          receptiveField = 17,
                          weightMapType = 0,
                          parallel = False, 
                          priviousResult = ''):

    logger = logging.getLogger(__name__)
    # print priviousResult, 1111111111
    assert len(patientsDirList) > 0

    # For more detail about weightMapType, place read the training cnofig file.
    if weightMapType == 0 or weightMapType == 1:
        foreBackRatio = 0.5
    elif weightMapType == 2:
        foreBackRatio = 0.75
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

    assert numOfSamplesPerPatient * numOfPatients + remainder == \
           numOfSamplesPerSubEpochTrain

    # If we use 3 patient and need 17 samples, 
    # the numOfSamplesPerPatientList will look like [6, 6, 5]
    numOfSamplesPerPatientList = [numOfSamplesPerPatient] * numOfPatients
    for idx in range(remainder):
        numOfSamplesPerPatientList[idx] += 1

    assert sum(numOfSamplesPerPatientList) == numOfSamplesPerSubEpochTrain, \
           '{}:{}'.format(sum(numOfSamplesPerPatientList), numOfSamplesPerSubEpochTrain)

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

    if parallel:
        pool = mp.Pool()
        results = []

    for idx, patientDir in enumerate(patientsDirList):

        numOfForeBackSamples = numOfForeBackSamplesList[idx]

        if parallel:
            result = pool.apply_async(getSamplesFromAPatient, args = (patientDir, 
                                                                      useROI, 
                                                                      modals, 
                                                                      normType, 
                                                                      trainSampleSize, 
                                                                      receptiveField, 
                                                                      weightMapType, 
                                                                      numOfForeBackSamples, 
                                                                      priviousResult))
            results.append(result)

        else:
            samplesOfAPatient, labelsOfAPatient = getSamplesFromAPatient(patientDir, 
                                                                         useROI, 
                                                                         modals, 
                                                                         normType, 
                                                                         trainSampleSize, 
                                                                         receptiveField, 
                                                                         weightMapType, 
                                                                         numOfForeBackSamples, 
                                                                         priviousResult)
            samplesList += samplesOfAPatient
            labelsList += labelsOfAPatient

    if parallel:
        pool.close()
        pool.join()

        for result in results:
            samplesOfAPatient, labelsOfAPatient = result.get()
            samplesList += samplesOfAPatient
            labelsList += labelsOfAPatient
    # ---------------------------------------------------------------------------------------

    # Release the memory.
    del samplesOfAPatient[:], samplesOfAPatient
    del labelsOfAPatient[:], labelsOfAPatient

    assert len(samplesList) == len(labelsList) == numOfSamplesPerSubEpochTrain
    assert samplesList[-1].shape[0] == len(modals) + int(priviousResult != ''), \
           'priviousResult: {}, samplesList[-1].shape: {}, modals: {}'.format(priviousResult, 
                                                                                 samplesList[-1].shape, 
                                                                                 modals)

    logger.info(logMessage('~', '~'))
    logger.info('Get all {} samples'.format(len(samplesList)))
    logger.info('The shape of a sample array equals: {}'.format(samplesList[0].shape))
    logger.info('The shape of a label array equals: {}'.format(labelsList[0].shape))
    logger.info(logMessage('~', '~'))

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



def getSamplesFromAPatient(patientDir, 
                           useROI, 
                           modals, 
                           normType, 
                           trainSampleSize, 
                           receptiveField, 
                           weightMapType, 
                           numOfForeBackSamples, 
                           priviousResult = ''):

    # Also store the ROI in the patientLabelArray if useROI == True
    # patientImageArray dtype = theano.config.floatX.
    # patientLabelArray dtype = int16

    loadedData = loadNiiData.loadSinglePatientData(patientDir, 
                                                   normType, 
                                                   modals, 
                                                   True,
                                                   useROI, 
                                                   priviousResult)

    patientImageArray, patientLabelArray, patientROIArray = loadedData

    assert patientImageArray.shape == (len(modals) + int(priviousResult != ''), 240, 240, 155)
    assert patientLabelArray.shape == (240, 240, 155)
    if useROI: assert patientROIArray.shape == (240, 240, 155)
    if not useROI: assert patientROIArray == []

    assert isinstance(patientImageArray, np.ndarray)
    assert isinstance(patientLabelArray, np.ndarray)

    samplesOfAPatient, labelsOfAPatient = sampleAPatient(patientImageArray, 
                                                         patientLabelArray, 
                                                         patientROIArray, 
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
    assert all([sample.shape == (len(modals) + int(priviousResult != ''), zSize, xSize, ySize) 
                for sample in samplesOfAPatient]), '{}'.format([s.shape 
                                                                for s in samplesOfAPatient])

    # For short the assert statements.
    rF = receptiveField
    assert len(labelsOfAPatient) == sum(numOfForeBackSamples)
    assert all([label.shape == (zSize - rF + 1, xSize - rF + 1, ySize - rF + 1)
                for label in labelsOfAPatient])

    # The two lists expend for each loop.
    return samplesOfAPatient, labelsOfAPatient



def sampleAPatient(patientImageArray, 
                   patientLabelArray, 
                   patientROIArray, 
                   numOfForeBackSamples,
                   useROI,
                   trainSampleSize,
                   receptiveField,
                   weightMapType):

    # logger = logging.getLogger(__name__)

    gTArray = patientLabelArray

    foreMask, backMask = getForeAndBackMask(patientLabelArray, 
                                            patientROIArray, 
                                            useROI,
                                            weightMapType)

    numOfForeSamples, numOfBackSamples = numOfForeBackSamples
    # Foreground samples coordinate range.
    # logger.debug('Get samples coordinates by foreMask.')
    foreSamplesCdList = getSamplesCoordinateList(foreMask, 
                                                 numOfForeSamples, 
                                                 trainSampleSize)
    # logger.debug('Get samples coordinates by backMask.')
    backSamplesCdList = getSamplesCoordinateList(backMask, 
                                                 numOfBackSamples, 
                                                 trainSampleSize)

    samplesCdList = foreSamplesCdList + backSamplesCdList

    samplesOfAPatient = getSamplesOfAPatient(patientImageArray, 
                                             samplesCdList,
                                             trainSampleSize)
    labelsOfAPatient, _ = getLabelsOfAPatient(gTArray, 
                                           samplesCdList, 
                                           receptiveField, 
                                           trainSampleSize)

    return samplesOfAPatient, labelsOfAPatient


def getSamplesOfAPatient(patientImageArray, samplesCdList, trainSampleSize):

    # logger = logging.getLogger(__name__)

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

    # logger = logging.getLogger(__name__)

    labelsOfAPatient = []
    labelSize = [axleSize - receptiveField + 1 
                 for axleSize in trainSampleSize]
    labelSize = tuple(labelSize)

    labelsCdList = []
    rFRadius = receptiveField / 2

    for sampleCd in samplesCdList:
        zLeft = sampleCd[0][0] + rFRadius
        zRight = sampleCd[0][1] - rFRadius
        xLeft = sampleCd[1][0] + rFRadius
        xRight = sampleCd[1][1] - rFRadius
        yLeft = sampleCd[2][0] + rFRadius
        yRight = sampleCd[2][1] - rFRadius

        labelsCdList.append([[zLeft, zRight], [xLeft, xRight], [yLeft, yRight]])
        
        if gTArray != []:
            labelsOfAPatient.append(gTArray[zLeft:zRight, xLeft: xRight, yLeft: yRight])
            assert labelsOfAPatient[-1].shape == labelSize, \
                   '{}:{}'.format(labelsOfAPatient[-1].shape, labelSize)

    return labelsOfAPatient, labelsCdList



def getSamplesCoordinateList(mask, numOfSamples, trainSampleSize):

    # logger = logging.getLogger(__name__)

    maskShape = mask.shape
    # If trainSampleSize = [25, 25, 25],
    # the centerLocOfSampleSize = [13, 13, 13]
    centerLocOfSampleSize = [int(math.ceil(axle / 2))
                             for axle in trainSampleSize]

    # logger.debug('centerLocOfSampleSize: {}'.format(centerLocOfSampleSize))
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
                       patientROIArray, 
                       useROI,
                       weightMapType):

    gTArray = patientLabelArray
    if useROI:
        # For brain ragion.
        ROIArray = patientROIArray
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

    assert foreMask.shape == (240, 240, 155)
    assert backMask.shape == (240, 240, 155)

    return foreMask, backMask



def sampleWholeBrain(patientDir, 
                     normType, 
                     modals, 
                     testSampleSize, 
                     receptiveField,
                     label = False, 
                     useROITest = True,
                     priviousResult = ''):

    loadedData = loadNiiData.loadSinglePatientData(patientDir, 
                                                   normType, 
                                                   modals, 
                                                   label,
                                                   useROITest, 
                                                   priviousResult)

    patientImageArray, patientLabelArray, patientROIArray = loadedData

    assert patientImageArray.shape == (len(modals) + int(priviousResult != ''), 240, 240, 155)
    assert isinstance(patientImageArray, np.ndarray)
    
    if label: assert patientLabelArray.shape == (240, 240, 155)
    if not label: assert patientLabelArray == []

    if useROITest: assert patientROIArray.shape == (240, 240, 155)
    if not useROITest: assert patientROIArray == []

    imageShape = list(patientImageArray[0].shape)

    if useROITest:
        ROIArray = patientROIArray
    else:
        ROIArray = np.ones(imageShape, dtype = 'int16')

    gTArray = patientLabelArray

    labelShape = [axle - receptiveField + 1 for axle in testSampleSize]
    labelShapeArray = np.asarray(labelShape, dtype = 'int16')
   
    wholeImageCoordList = getWholeImageCoord(imageShape,
                                             ROIArray,
                                             testSampleSize,
                                             labelShape)

    samplesOfWholeImage = getSamplesOfAPatient(patientImageArray, 
                                               wholeImageCoordList,
                                               testSampleSize)

    labelsOfWholeImage, wholeLabelCoordList = getLabelsOfAPatient(gTArray, 
                                                                  wholeImageCoordList, 
                                                                  receptiveField, 
                                                                  testSampleSize)
    if label:
        assert len(samplesOfWholeImage) == len(labelsOfWholeImage) == len(wholeLabelCoordList)

    return samplesOfWholeImage, labelsOfWholeImage, wholeLabelCoordList, imageShape, gTArray



def getWholeImageCoord(imageShape,
                       ROIArray,
                       testSampleSize,
                       labelShape):

    wholeImageCoordList = []
    zMinNext = 0

    zAxleFinished = False

    while not zAxleFinished:
        zMax = min(zMinNext + testSampleSize[0], imageShape[0])
        zMin = zMax - testSampleSize[0]
        zMinNext = zMinNext + labelShape[0]

        if zMax < imageShape[0]:
            zAxleFinished = False
        else:
            zAxleFinished = True


        xMinNext = 0
        xAxleFinished = False

        while not xAxleFinished:
            xMax = min(xMinNext + testSampleSize[1], imageShape[1])
            xMin = xMax - testSampleSize[1]
            xMinNext = xMinNext + labelShape[1]

            if xMax < imageShape[1]:
                xAxleFinished = False
            else:
                xAxleFinished = True


            yMinNext = 0
            yAxleFinished = False

            while not yAxleFinished:
                yMax = min(yMinNext + testSampleSize[2], imageShape[2])
                yMin = yMax - testSampleSize[2]
                yMinNext = yMinNext + labelShape[2]

                if yMax < imageShape[2]:
                    yAxleFinished = False
                else:
                    yAxleFinished = True

                if not np.any(ROIArray[zMin:zMax, xMin:xMax, yMin:yMax]):
                    continue

                assert yMax - yMin == testSampleSize[2]
                assert yMin >= 0 and yMax <= imageShape[2]

                assert xMax - xMin == testSampleSize[1]
                assert xMin >= 0 and xMax <= imageShape[1]

                assert zMax - zMin == testSampleSize[0]
                assert zMin >= 0 and zMax <= imageShape[0]

                wholeImageCoordList.append([[zMin, zMax], [xMin, xMax], [yMin, yMax]])

    return wholeImageCoordList






    