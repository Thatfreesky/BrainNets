import nibabel as nib
import numpy as np
import os
import logging
import shutil
import theano
import time
import pickle
from datetime import datetime
from random import shuffle
import multiprocessing as mp
from nipype.interfaces.ants import N4BiasFieldCorrection
from medpy.filter import IntensityRangeStandardization


import general as ge
from general import logMessage, logTable



def preProcessingWithN4(inputDir,
                        modals = ['t1ce', 't1'],
                        bsplineFittingDistance = 200,
                        shrinkFactor = 2,
                        iterations = [20,20,20,10],
                        parallel = False):

    '''
    # The example from nipype documents(https://pythonhosted.org/nipype/interfaces/generated/nipype.interfaces.ants.segmentation.html)

    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = 'structural.nii'
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50,50,30,20]
    n4.cmdline 


    # The parameters from paper: Deep Convolutional Neural Networks for the Segmentation of Gliomas in Multi-sequence MRI

    n_iterations = [20,20,20,10]
    shrink_factor = 2
    bspline_fitting_distance = 200

    '''

    if parallel:
        cpus = mp.cpu_count()
        pool = mp.Pool(processes = cpus)


    for patientDir, modalFileName in ge.goToTheImageFiles(inputDir):

        modalFileNameSegList = modalFileName.split('_')
        modalNameWithFileType = modalFileNameSegList[-1]

        modalName = modalNameWithFileType.split('.')[0]

        assert modalName in ['flair', 't2', 't1ce', 't1', 'seg']

        if modalName in modals:

            inputImagePath = os.path.join(patientDir, modalFileName)
            outputImagePath = os.path.join(patientDir, 'N4' + modalFileName)

            if parallel:
                pool.apply_async(N4BiasCorrectAFile, args = (inputImagePath,
                                                             bsplineFittingDistance,
                                                             shrinkFactor,
                                                             iterations,
                                                             outputImagePath))
            else:
                N4BiasCorrectAFile(inputImagePath,
                                   bsplineFittingDistance,
                                   shrinkFactor,
                                   iterations,
                                   outputImagePath)

    if parallel:
        pool.close()
        pool.join()


def N4BiasCorrectAFile(inputImagePath,
                       bsplineFittingDistance,
                       shrinkFactor,
                       iterations,
                       outputImagePath):

    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = inputImagePath
    n4.inputs.bspline_fitting_distance = bsplineFittingDistance
    n4.inputs.shrink_factor = shrinkFactor
    n4.inputs.n_iterations = iterations
    n4.inputs.output_image = outputImagePath

    n4.run()  


def transIntensityRangesStand(trainedModel,
                              inputDir = ''):

    '''
    Reference

    https://pythonhosted.org/MedPy/generated/medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization.html#medpy-filter-intensityrangestandardization-intensityrangestandardization
    
    Examples

    We have a number of similar images with varying intensity ranges. To make them comparable, we would like to transform them to a common intensity space. Thus we run:

    >>>
    >>> from medpy.filter import IntensityRangeStandardization
    >>> irs = IntensityRangeStandardization()
    >>> trained_model, transformed_images = irs.train_transform(images)

    Let us assume we now obtain another, new image, that we would like to make comparable to the others. As long as it does not differ to much from these, we can simply call:

    >>>
    >>> transformed_image = irs.transform(new_image)

    For many application, not all images are already available at the time of execution. It would therefore be good to be able to preserve a once trained model. The solution is to just pickle the once trained model:

    >>>
    >>> import pickle
    >>> with open('my_trained_model.pkl', 'wb') as f:
    >>>     pickle.dump(irs, f)

    And load it again when required with:

    >>>
    >>> with open('my_trained_model.pkl', 'r') as f:
    >>>     irs = pickle.load(f)

    '''

    logger = logging.getLogger(__name__)

    with open(trainedModel, 'r') as f:

        irs, _, tableRowList = pickle.load(f)

    if inputDir == '': inputDir = tableRowList['inputDir']

    grade = tableRowList['grade']
    modal = tableRowList['modal']
    afterN4 = tableRowList['afterN4']
    landmark = tableRowList['landmark']
    stdrange = tableRowList['stdrange']

    for patientDir, modalFileName in ge.goToTheImageFiles(inputDir, grade):

        if afterN4 and 'N4' not in modalFileName: continue
        if not afterN4 and 'N4' in modalFileName: continue

        modalFileNameSegList = modalFileName.split('_')
        modalNameWithFileType = modalFileNameSegList[-1]

        modalName = modalNameWithFileType.split('.')[0]

        assert modalName in ['flair', 't2', 't1ce', 't1', 'seg']

        if modalName == modal:
            imagePath = os.path.join(patientDir, modalFileName)

            image = nib.load(imagePath)
            imageArray = image.get_data().astype(theano.config.floatX)

            maskArray = imageArray > 0
            imageArrayAfterWithMask = imageArray[maskArray]
            transformedArray = irs.transform(imageArrayAfterWithMask)

            standardArray = imageArray
            standardArray[maskArray] = transformedArray

            assert np.all(standardArray == imageArray)
            
            standardImage = nib.Nifti1Image(standardArray, image.affine)
            standardImage.set_data_dtype(theano.config.floatX)

            standardImageName = '{}_{}'.format(landmark, stdrange) + modalFileName
            standardImageNameWithPath = os.path.join(patientDir, standardImageName)
            nib.save(standardImage, standardImageNameWithPath)



def transIntensityRangesStandParallel(modelsDir):

    cpus = mp.cpu_count()
    pool = mp.Pool(processes = cpus)

    modelsPathList = [os.path.join(modelsDir, modelName) for modelName in os.listdir(modelsDir)]

    for model in modelsPathList:

        pool.apply_async(transIntensityRangesStand, args = (model, ''))

    pool.close()
    pool.join()




def trainIntensityRangeStandModel(grade,
                                  modal,
                                  numUsedToTrain,
                                  inputDir,
                                  storeLocation,
                                  afterN4,
                                  cutoff = (1,99),
                                  landmark = [25, 50, 75],
                                  stdrange = 'auto'):

    logger = logging.getLogger(__name__)

    casesList = []

    for patientDir, modalFileName in ge.goToTheImageFiles(inputDir, grade):

        if afterN4 and 'N4' not in modalFileName: continue
        if not afterN4 and 'N4' in modalFileName: continue

        modalFileNameSegList = modalFileName.split('_')
        modalNameWithFileType = modalFileNameSegList[-1]

        modalName = modalNameWithFileType.split('.')[0]

        assert modalName in ['flair', 't2', 't1ce', 't1', 'seg']

        if modalName == modal:
            imagePath = os.path.join(patientDir, modalFileName)
            casesList.append(imagePath)

    assert casesList != []

    shuffle(casesList)

    if numUsedToTrain == -1: 
        casesUsedToTrain = casesList[:]
        logger.info('Use all cases to train')
    elif numUsedToTrain > len(casesList):
        casesUsedToTrain = casesList[:]
        logger.info('There are only {} cases, {} supply, \
                    so we use all cases to train'.format(len(casesList, numUsedToTrain)))
    else:
        casesUsedToTrain = casesList[:numUsedToTrain]
        logger.info('Use {} cases to train'.format(numUsedToTrain))

    imagesArrayList = []
    masksArrayList = []

    for imagePath in casesUsedToTrain:

        image = nib.load(imagePath)

        imageArray = image.get_data().astype(theano.config.floatX)
        maskArray = imageArray > 0

        imagesArrayList.append(imageArray)
        masksArrayList.append(maskArray)


    logger.info('Load all training data')

    irs = IntensityRangeStandardization(cutoff, landmark, stdrange)
    logger.info('The model parameters are:')

    tableRowList = []
    tableRowList.append(['-', '-'])
    tableRowList.append(['grade', grade])
    tableRowList.append(['modal', modal])
    tableRowList.append(['numUsedToTrain', numUsedToTrain])
    tableRowList.append(['inputDir', inputDir])
    tableRowList.append(['storeLocation', storeLocation])
    tableRowList.append(['afterN4', afterN4])
    tableRowList.append(['cutoff', cutoff])
    tableRowList.append(['landmark', landmark])
    tableRowList.append(['stdrange', stdrange])
    
    tableRowList.append(['-', '-'])

    logger.info(logTable(tableRowList))

    logger.info('Begin training intensity range standardization model')


    startTime = time.time()
    irs.train([imageArr[maskArr] for imageArr, maskArr in zip(imagesArrayList, masksArrayList)])
    endTime = time.time() - startTime

    logger.info('Trained the model, which taken {} seconds'.format(endTime))

    modelName = '{}_{}_{}_{}_{}_{}.pkl'.format(grade, modal, afterN4, numUsedToTrain, landmark, stdrange)
    modelNameWithPath = os.path.join(storeLocation, modelName)

    with open(modelNameWithPath, 'wb') as f:
        pickle.dump([irs, casesUsedToTrain, dict(tableRowList)], f)



def trainIRSMsParallel(privateParaList, commonParaTuple):

    storeLocationInTimeSegList = [str(item)for item in commonParaTuple[2:]]

    storeLocation = '_'.join(storeLocationInTimeSegList)
    storeLocation += '_{}_{}'.format(privateParaList[0][-1], privateParaList[-1][-1])
    timeString = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    storeLocationInTime = storeLocation + '_' + timeString

    fullStoreLocationInTime = os.path.join(commonParaTuple[1], storeLocationInTime)

    os.mkdir(fullStoreLocationInTime)

    commonParaList = list(commonParaTuple)
    commonParaList[1] = fullStoreLocationInTime

    commonParaTuple = tuple(commonParaList)

    cpus = mp.cpu_count()
    pool = mp.Pool(processes = cpus)

    for model in privateParaList:

        pool.apply_async(trainIntensityRangeStandModel, args = model + commonParaTuple)

    pool.close()
    pool.join()












    
















    

    