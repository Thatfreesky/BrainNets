import nibabel as nib
import numpy as np
import theano
import logging
import os


def loadSinglePatientData(patientDir, normType = 'normImage', modals, label = True, ROI = True):

    logger = logging.getLogger(__name__)

    assert os.path.isdir(patientDir)

    modalsDict = {'tice':0, 't1':1, 't2':2, 'flair':3}

    imageArrayList = []
    labelArray = []
    ROIArray = []

    modalsPathList = []
    labelPath = ''
    ROIPath = ''

    for fileName in os.listdir(patientDir):

        if fileName.startswith(normType):
            modalsPathList.append(os.join(patientDir, fileName))
        
        if 'seg' in fileName:
            labelPath = os.join(patientDir, fileName)

        if 'ROI' in fileName:
            ROIPath = os.join(patientDir, fileName)

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

        imageArrayList.append(readArray(filePath))

    imageArray = np.asarray(imageArrayList, dtype = theano.floatX)

    assert len(imageArrayList) == len(modals)

    if label:
        assert labelPath != ''
        labelArray = readArray(labelPath)

    if ROI:
        assert ROIPath != ''
        ROIArray = readArray(ROIPath)


    return imageArray, labelArray, ROIArray



def readArray(filePath):

    logger = logging.getLogger(__name__)

    assert filePath.endswith('.nii.gz')

    image = nib.load(filePath)
    imageArray = image.get_data()

    return imageArray

    
