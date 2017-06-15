'''
This module is designed to creat the ROI mask for BraTS data.
For the BraTS, it is easy to achieve this task.
Only set the value to 1 at the region that the Brats images is nonzero
'''


import SimpleITK as sitk
import numpy as np
import os
import logging
import shutil
import theano

import general

def normAndCreateROIForAllFiles(dataPath, outputPath, forTestData = False):

    '''
    For the Brats 2015 training data, we should assign the path of BRATS2015_Training
    to the dataPath parameter.
    The dictionary tree of BRATS2015_Training looks like:
    BRATS2015_Training-----------HGG------brats_2013_pat0001_1----VSD.Brain.XX.O.MR_Flair.54512--VSD.Brain.XX.O.MR_Flair.54512.mha
                      \             \                         \
                       \             \                         \--......
                        \             \                         .
                         \             \--......                 .
                          \             .                         .
                           \             .
                            \             .
                             \
                              \--LGG--------------------------......

    We will creat ROI files for every patient and organize the new ROI and the original files in such structure:
    outputPath-------- -HGG------brats_2013_pat0001_1----VSD.Brain.XX.O.MR_Flair.54512.mha
              \            \                         \
               \            \                         \--ROI
                \            \--......                 \
                 \            .                         \......
                  \            .                         .
                   \            .                         .
                    \                                      . 
                     \--LGG--------------------------......

    We will provide two type of normalization results. 
    One for norm the whole image, one for norm the nonzero region.

    '''

    logger = logging.getLogger(__name__)

    ROIDir = outputPath
    general.makeDir(ROIDir)

    for gradeDirItem in os.listdir(dataPath):

        gradeDir = os.path.join(dataPath, gradeDirItem)
        ROIGradeDir = os.path.join(ROIDir, gradeDirItem)
        general.makeDir(ROIGradeDir)

        for patientDirItem in os.listdir(gradeDir):

            patientDir = os.path.join(gradeDir, patientDirItem)
            ROIPatientDir = os.path.join(ROIGradeDir, patientDirItem)
            general.makeDir(ROIPatientDir)

            haveCreatedROIFile = False

            folderNum = len(os.listdir(patientDir))
            if (forTestData and folderNum != 4) or (not folderNum and folderNum != 5):
                print patientDir

            for modalDirItem in os.listdir(patientDir):

                modalDir = os.path.join(patientDir, modalDirItem)

                modalFileList = [fileItem for fileItem in os.listdir(modalDir) 
                                 if fileItem.endswith('.mha') or fileItem.endswith('.nii')]
                assert len(modalFileList) == 1, (patientDir, modalFileList)

                modalFileName = modalFileList[0]
                if modalFileName.endswith('.nii'):
                    print patientDir
                    continue

                # Althrough we create the normalization file, but thet are so large compare to 
                # the orgimal file and. I guess we can not read all files in memory. So, we also
                # copy the orginal files, for we may consider we may read all files in memory and
                # norm it on the fly.

                modalFileNameWithPath = os.path.join(modalDir, modalFileName)
                shutil.copy2(modalFileNameWithPath, ROIPatientDir)

                # Be careful, do not norm the ground truth file.

                if 'OT' == modalFileName.split('.')[-3]:
                    logger.debug(modalFileName.split('.'))
                    continue
                
                normAndCreateROIForOneFile(modalFileNameWithPath, ROIPatientDir, haveCreatedROIFile)
                    # TODO. Wheather the ROI files created by different modals are same?
                haveCreatedROIFile = True

            if forTestData: 
                assert len(os.listdir(ROIPatientDir)) == 13
            else:
                assert len(os.listdir(ROIPatientDir)) == 14

        logger.info('Create ROI and normalize {} {} patients data files'.format(len(os.listdir(gradeDir)), gradeDirItem))


def normAndCreateROIForOneFile(modalFileNameWithPath, ROIPatientDir, haveCreatedROIFile):
    '''
    Here we provide two kinds of normalizition, meanwhile we create the ROI for each patient.

    Norm the whole image.
            |------------|
            |            |
            |            |
            |            |
            |            |
            |            |
            |            |
            |            |
            |------------|

    Norm the brain region.

            |------------|
            |  .------.  |
            | /        \ |
            | |        | |
            | |        | |
            | |        | |
            |  \      /  |
            |   \____/   |      |
            |------------|

    '''

    logger = logging.getLogger(__name__)

    image = sitk.ReadImage(modalFileNameWithPath)
    imageArray = sitk.GetArrayFromImage(image)

    # It's easy to create the ROI mask. Just choose the nonzero region as ROI.

    if not haveCreatedROIFile:

        ROIBoolArray = imageArray > 0
        ROIArray = ROIBoolArray.astype('int16')
        ROIImage = sitk.GetImageFromArray(ROIArray)

        ROIPatientDirSegList = ROIPatientDir.split('/')
        # Get the patient name
        patientName = ROIPatientDirSegList[-1]

        # Name the ROI file
        ROIFileName = patientName + '.ROI.mha'
        ROIFileNameWithPath = os.path.join(ROIPatientDir, ROIFileName)
        sitk.WriteImage(ROIImage, ROIFileNameWithPath)


    modalFileNameWithPathSegList = modalFileNameWithPath.split('/')
    normFileBaseName = modalFileNameWithPathSegList[-1]

    assert normFileBaseName.endswith('.mha')

    imageStd = np.std(imageArray)

    # Norm the whole image.
    if imageStd == 0:
        normImageArray = imageArray
    else:
        normImageArray = (imageArray - np.mean(imageArray)) / np.std(imageArray)

    normImageArray = normImageArray.astype(theano.config.floatX)

    normImage = sitk.GetImageFromArray(normImageArray)
    normImageName = 'normImage' +  normFileBaseName
    normImageNameWithPath = os.path.join(ROIPatientDir, normImageName)
    sitk.WriteImage(normImage, normImageNameWithPath)

    # logger.debug('Saved the normImage, {}'.format(normImageNameWithPath))

    # Only need to consider the nonzero value, so we extract those nonzero elements.
    brainBoolArray = imageArray > 0
    brainElementVector = imageArray[brainBoolArray]
    brainMean = np.mean(brainElementVector)
    brainStd = np.std(brainElementVector)

    # Norm the brain region.
    if brainStd == 0:
        normBrainArray = imageArray
    else:
        brainMeanArray = brainBoolArray * brainMean
        normBrainArray = (imageArray - brainMeanArray) / brainStd

    normBrainArray = normBrainArray.astype(theano.config.floatX)

    normBrain = sitk.GetImageFromArray(normBrainArray)
    normBrainName = 'normBrain' +  normFileBaseName
    normBrainNameWithPath = os.path.join(ROIPatientDir, normBrainName)
    sitk.WriteImage(normBrain, normBrainNameWithPath)

    # logger.debug('Saved the normBrain, {}'.format(normBrainNameWithPath))
