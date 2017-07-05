import nibabel as nib
import numpy as np
import theano
import logging
import os

def loadSinglePatientData(patientDir, useROI, modals, normType, forTestData = False):

    logger = logging.getLogger(__name__)

    assert os.path.isdir(patientDir)

    imageNameList, labelNameList = findImgAndLabelFileNameList(patientDir, 
                                                               useROI, 
                                                               modals, 
                                                               normType,
                                                               forTestData)

    imageNameWithPathList = [os.path.join(patientDir, imageName) 
                             for imageName in imageNameList]
    
    imageArrayList = [readImageArray(imageNameWithPath) 
                      for imageNameWithPath in imageNameWithPathList]

    patientImageArray = np.asarray(imageArrayList, dtype = theano.config.floatX)
    del imageArrayList[:], imageArrayList
    assert patientImageArray.shape == (len(modals), 155, 240, 240)

    if len(labelNameList) == 0:
        assert forTestData and not useROI
        return patientImageArray, []


    labelNameWithPathList = [os.path.join(patientDir, labelName)
                             for labelName in labelNameList]

    labelArrayList = [readImageArray(labelNameWithPath) 
                      for labelNameWithPath in labelNameWithPathList]

    patientLabelArray = np.asarray(labelArrayList, dtype = 'int32')
    del labelArrayList[:], labelArrayList
    assert patientLabelArray.shape == (int(not forTestData) + int(useROI), 155, 240, 240)

    return patientImageArray, patientLabelArray


def findImgAndLabelFileNameList(patientDir, useROI, modals, normType, forTestData):

    logger = logging.getLogger(__name__)

    modalsDict = {'T1c':0, 'T1':1, 'T2':2, 'Flair':3}

    imageNameList = []
    labelNameList = []

    for item in os.listdir(patientDir):
        if item.endswith('txt'): 
            continue

        assert item.endswith('.mha'), (item, patientDir)

        if normType == 0:
            prefix  = 'normImage'
        else:
            prefix = 'normBrain'

        # itemNameSegList looks like 
        # ['normImageVSD', 'Brain', 'XX', 'O', 'MR_T1', '54591', 'mha']
        itemNameSegList = item.split('.')
        assert len(itemNameSegList) > 2


        if itemNameSegList[0].startswith(prefix):

            modalName = findModalName(item)

            if modalName in modals:
                imageNameList.append(item)
                continue

        if itemNameSegList[-2] == 'ROI' and useROI:
            assert len(itemNameSegList) == 3
            # Use labelNameList to store ROI file name if useROI == True.
            labelNameList.append(item)
            continue

        if itemNameSegList[-3] == 'OT' and not forTestData:
            labelNameList.append(item)



    assert len(imageNameList) == len(modals), '{} == {}'.format(imageNameList, modals)
    assert len(labelNameList) == int(not forTestData) + int(useROI), \
                                '{} == {}'.format(labelNameList, int(not forTestData) + int(useROI))

    # This two lists should contain different item.
    assert set(imageNameList) & set(labelNameList) == set()

    # Now, we should to reorder the image and label name list.
    # First, assign its right order number.
    reorderImageNameList = [[modalsDict[findModalName(fileName)], fileName] 
                             for fileName in imageNameList]
    # Reorder it via its right order number assigned at last step.
    reorderImageNameList.sort(key = lambda pair: pair[0])
    # Release the assigned number.
    reorderImageNameList = [pair[1] for pair in reorderImageNameList]

    # For logger.
    imgShortNameLs = [findModalName(fileName) for fileName in imageNameList]
    reorderImgShortNameLs = [findModalName(fileName) for fileName in reorderImageNameList]
    logger.debug('We reorder the imageNameList from {} to {}.'.format(imgShortNameLs, 
                                                                     reorderImgShortNameLs))
    imageNameList = reorderImageNameList

    # Now turn to the label name list.
    if not useROI:
        assert len(labelNameList) == int(not forTestData)
    elif not forTestData:
        # First, we extract the ground truth file name
        assert len(labelNameList) == 2
        groundTruthList = [fileName 
                           for fileName in labelNameList 
                           if fileName.split('.')[-3] == 'OT']
        # Second, we extract the ROI file name
        ROIList = [fileName 
                   for fileName in labelNameList 
                   if fileName.split('.')[-2] == 'ROI']
        assert len(groundTruthList) == 1
        assert len(ROIList) == 1

        # By this way, we can ensure the ground truth file is at the first place.
        labelNameList = groundTruthList + ROIList

    return imageNameList, labelNameList


def findModalName(fileName):

    logger = logging.getLogger(__name__)

    modalsDict = {'T1c':0, 'T1':1, 'T2':2, 'Flair':3}

    fileNameSegList = fileName.split('.')
    modalName = fileNameSegList[-3]
    modalName = modalName[3:]

    assert modalName in modalsDict.keys()

    return modalName


def readImageArray(filePath):

    logger = logging.getLogger(__name__)

    # Now, only consider read the mha file.
    assert filePath.endswith('.mha')

    image = sitk.ReadImage(filePath)
    imageArray = sitk.GetArrayFromImage(image)

    assert imageArray.shape == (155, 240, 240)

    return imageArray
