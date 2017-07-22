import nibabel as nib
import numpy as np
import theano
import logging
import os

from general import numpyArrayCounter


def loadSinglePatientData(patientDir, 
                          normType = 'normImage', 
                          modals = ['t1ce', 't1', 't2', 'flair'], 
                          label = True, 
                          ROI = True, 
                          priviousResult = ''):

    logger = logging.getLogger(__name__)

    assert os.path.isdir(patientDir)

    modalsDict = {'t1ce':0, 't1':1, 't2':2, 'flair':3}

    imageArrayList = []
    patientLabelArray = []
    ROIArray = []

    modalsPathList = []
    labelPath = ''
    ROIPath = ''

    for fileName in os.listdir(patientDir):

        if fileName.startswith(normType):
            modalsPathList.append(os.path.join(patientDir, fileName))
        
        if 'seg' in fileName:
            labelPath = os.path.join(patientDir, fileName)

        if 'ROI' in fileName:
            ROIPath = os.path.join(patientDir, fileName)

    finalList = [''] * 4

    for filePath in modalsPathList:

        filePathSegList = filePath.split('.')
        modalSeg = filePathSegList[-3]
        modalSeg = modalSeg.split('_')[-1]

        if modalSeg not in modals: continue

        modalFig = modalsDict[modalSeg]
        finalList[modalFig] = filePath

    for idx, filePath in enumerate(finalList):

        if filePath == '':
            continue

        imageArrayList.append(readArray(filePath).astype(theano.config.floatX))

    if priviousResult != '':
        patientName = patientDir.split('/')[-1]
        priviousImage = os.path.join(priviousResult, patientName + '.nii.gz')
        imageArrayList.append(readArray(filePath).astype(theano.config.floatX))

    patientImageArray = np.asarray(imageArrayList, dtype = theano.config.floatX)

    assert len(imageArrayList) == len(modals), '{}, {}'.format(len(imageArrayList), modals)

    if label:
        assert labelPath != ''
        patientLabelArray = readArray(labelPath).astype('int16')
        oldLabelCount = numpyArrayCounter(patientLabelArray)
        if len(oldLabelCount) != 4: logger.debug(oldLabelCount)

        temArray = (patientLabelArray == 4).astype(int)
        temArray *= -1
        patientLabelArray += temArray

        newLabelCount = numpyArrayCounter(patientLabelArray)

        assert np.all(oldLabelCount[:-1] == newLabelCount[:-1])
        if oldLabelCount[-1][0] == 4: assert newLabelCount[-1][0] == 3
        assert oldLabelCount[-1][1] == newLabelCount[-1][1], (oldLabelCount, newLabelCount)



    if ROI:
        assert ROIPath != ''
        ROIArray = readArray(ROIPath).astype('int16')


    return patientImageArray, patientLabelArray, ROIArray



def readArray(filePath):

    logger = logging.getLogger(__name__)

    assert filePath.endswith('.nii.gz')

    image = nib.load(filePath)
    imageArray = image.get_data()

    return imageArray

    
