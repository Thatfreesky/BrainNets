import nibabel as nib
import numpy as np
import theano
import logging
import os


def loadSinglePatientData(patientDir, 
                          normType = 'normImage', 
                          modals = ['t1ce', 't1', 't2', 'flair'], 
                          label = True, 
                          ROI = True):

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

        imageArrayList.append(readArray(filePath))

    patientImageArray = np.asarray(imageArrayList, dtype = theano.config.floatX)

    assert len(imageArrayList) == len(modals), '{}, {}'.format(len(imageArrayList), modals)

    if label:
        assert labelPath != ''
        patientLabelArray = readArray(labelPath)

    if ROI:
        assert ROIPath != ''
        ROIArray = readArray(ROIPath)


    return patientImageArray, patientLabelArray, ROIArray



def readArray(filePath):

    logger = logging.getLogger(__name__)

    assert filePath.endswith('.nii.gz')

    image = nib.load(filePath)
    imageArray = image.get_data()

    return imageArray

    
