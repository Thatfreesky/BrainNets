import os
import glob
import time
import types
import bisect
import random
import shutil
import theano
import logging
import pyprind
# import copy_reg
import ipyvolume
import numpy as np
from time import sleep
import SimpleITK as sitk
import multiprocessing as mp
from collections import Counter
from IPython.display import display
import pathos.multiprocessing as pmp
from tqdm import tqdm_notebook, tnrange

random.seed(1321)
np.random.seed(1321)

def cube2Vector(cubeImage):
    return list(cubeImage.flatten())

def vector2Cube(vectorImage, cubeSize):
    return np.array(vectorImage).reshape((cubeSize, cubeSize, cubeSize))

def ndarrayCounter(ndarray):
    unique, counts = np.unique(ndarray, return_counts = True)
    return np.asarray((unique, counts)).T

def show3D(filePath):
    image = sitk.ReadImage(filePath)
    img = sitk.GetArrayFromImage(image)
    
    display(ipyvolume.volume.quickvolshow(img))

def makeDir(dir, force = False):
    logger = logging.getLogger(__name__)
    assert type(dir) == str,'The type of dir argument should be a String.'

    if not os.path.exists(dir):
        os.mkdir(dir)
    elif force:
        shutil.rmtree(dir)
        os.mkdir(dir)
    elif os.path.exists(dir) and not force:
        logger.debug(dir + 'already exists')
        
def findModalDir(upDir, modal):
    for modalDirItem in os.path.listdir(upDir):
        modalDirItemSegment = modalDirItem.split('.')
        if 'MR_' + modal in modalDirItemSegment:
            return modalDirItem
    else:
        print 'No such modal'


def normalizeData(modalFileNameWithPath, 
                  normModalFileNameWithPath, 
                  clipScope = (0.5, 99.5)):
    if 'OT' in os.path.basename(modalFileNameWithPath):
        shutil.copyfile(modalFileNameWithPath, normModalFileNameWithPath)
        assert os.path.isfile(normModalFileNameWithPath)

        return

    image3D = sitk.ReadImage(modalFileNameWithPath)
    image3DArray = sitk.GetArrayFromImage(image3D)

    bottom, top = np.percentile(image3DArray, clipScope)

    clipedImage3DArray = np.clip(image3DArray, bottom, top)

    standardDeviation = np.std(clipedImage3DArray)

    if standardDeviation != 0:
        clipedImage3DArray = (clipedImage3DArray - np.mean(clipedImage3DArray)) / standardDeviation

    clipedImage3D = sitk.GetImageFromArray(clipedImage3DArray)

    sitk.WriteImage(clipedImage3D, normModalFileNameWithPath)



def normalizeDataSet(dataPath = '../data/BRATS2015_Training/', 
                     normalizedDataDir = '../data/normalizedDataSet/', 
                     clipScope = (0.5, 99.5), 
                     parallel = False):

    '''
    Normalizes all models data of all patients excluding ground truth.
    First, clips top and bottom one percent of pixel intensities
    Then, subtracts mean and div by std dev for each volumn.
    '''
    logger = logging.getLogger(__name__)

    startTime = time.time()

    normDataDir = normalizedDataDir
    makeDir(normDataDir)

    logger.debug('normDataDir: {}'.format(normDataDir))

    if parallel:
        logger.info('Using multiprocessing')
        cpuCount = mp.cpu_count()
        pool = mp.Pool(processes = cpuCount)


    for gradeDirItem in os.listdir(dataPath):

        logger.debug('gradeDirItem: {}'.format(gradeDirItem))

        gradeDir = os.path.join(dataPath, gradeDirItem)
        normGradeDir = os.path.join(normDataDir, gradeDirItem)
        makeDir(normGradeDir)

        for patientDirItem in os.listdir(gradeDir):

            patientDir = os.path.join(gradeDir, patientDirItem)
            normPatientDir = os.path.join(normGradeDir, patientDirItem)
            makeDir(normPatientDir)

            for modalDirItem in os.listdir(patientDir):

                modalDir = os.path.join(patientDir, modalDirItem)
                normModalDir = os.path.join(normPatientDir, modalDirItem)
                makeDir(normModalDir)

                modalFileList = [fileItem for fileItem in os.listdir(modalDir) if fileItem.endswith('.mha')]
                logger.debug('modalFileList: {}'.format(modalFileList))

                assert len(modalFileList) == 1

                modalFileName = modalFileList[0]

                modalFileNameWithPath = os.path.join(modalDir, modalFileName)
                normModalFileNameWithPath = os.path.join(normModalDir, modalFileName)

                logger.debug('modalFileNameWithPath: {}'.format(modalFileNameWithPath))
                logger.debug('normModalFileNameWithPath: {}'.format(normModalFileNameWithPath))

                if parallel:
                    pool.apply_async(normalizeData, args = (modalFileNameWithPath, 
                                                            normModalFileNameWithPath, 
                                                            clipScope))
                    
                else:
                    normalizeData(modalFileNameWithPath, normModalFileNameWithPath, clipScope)

    if parallel:
        pool.close()
        pool.join()


    logger.info('The time for normalizing all data is {}'.format(time.time() - startTime))


            
def makeCubes(subStructure = 'edema', 
              modal = 'T2', 
              dataPath = '../data/BRATS2015_Training/', 
              cubeDirectory = '../data/cubeData',
              grade = 'HGG', 
#               numberOfPatients = 10, 
#               numberOfPointsPerPatient = 100, 
              cubeSize = 5, 
              valDataRatio = 0.1):
    '''
    According to the size of receptive field of network, generate the data points, i.e., the cubeSize should equal the size of receptive field
    Every generated data point have two part, the 3D data array with shape (cubeSize, cubeSize, cubeSize) and the its label
    The data label use a int number to represent, i.e., 1 for necrosis, 2 for edma, 3 for non-enhancing, 4 for enhancing, 0 for everything else
    -----------------
    subStructure: tumer substructure, can choose edema, solid, necrotic or non-enhancing. Note, not every patient contains all kind of substruce
    modal: a modal list. Choose different modal for different substructure,for example, T2 and FLAIR are usually used to segment edema
    dataPath: path to the BRATS data
    grade: choose different grade patient, such as HGG or LGG
    numberOfPatients: the number of patients that are used to generate data point
    numberOfPoints: the number of generated data points
    cubeSize: the shape of cube like 3D array, must be an odd number
    -----------------
    '''
    assert subStructure in ['edema', 'necrosis', 'non-enhancing', 'enhancing'], '''{} not in the list ['edema', 'necrosis', 'non-enhancing', 'enhancing']'''.format(subStrcture)
#     assert False not in [mod in ['T1', 'T1c', 'T2', 'FLAIR'] for mod in modal], '''{} is not the subset of ['T1', 'T1c', 'T2', 'FLAIR']'''.format(modal)
    assert modal in ['T1', 'T1c', 'T2', 'FLAIR'], '''{} not in the list ['T1', 'T1c', 'T2', 'FLAIR']'''.format(modal)
    assert grade in ['HGG', 'LGG'], '''{} not in the list ['HGG', 'LGG']'''.format(grade)
    assert cubeSize % 2 == 1 and cubeSize > 0, '''{} is not a odd number or {} not more than 0'''.format(cubeSize, cubeSize)
    
    logger = logging.getLogger(__name__)

    startTime = time.time()

    makeDir(cubeDirectory)
    gradeCubeDir = os.path.join(cubeDirectory, grade)
    makeDir(gradeCubeDir)
    # The cubeData directory tree is simular with the BRSTS2015_Training directory tree
    gradeDir = os.path.join(dataPath, grade)
    patientDirList = os.listdir(gradeDir)
    for patientDirItem in tqdm_notebook(patientDirList, desc = 'For every patient'):
        patientCubeDir = os.path.join(gradeCubeDir, patientDirItem)
        makeDir(patientCubeDir)
        
        patientDir = os.path.join(gradeDir, patientDirItem)
        modalDirList = os.listdir(patientDir)
        
        targetModalDirItem = ''
        groundTruthDirItem = ''
        for modalDirItem in modalDirList:
            modalDirItemSegment = modalDirItem.split('.')
            if 'MR_' + modal in modalDirItemSegment:
                targetModalDirItem = modalDirItem
            
            if 'OT' in modalDirItemSegment:
                groundTruthDirItem = modalDirItem
        
        assert targetModalDirItem != '', 'Can not find the directory of this modal:' + modal
        assert groundTruthDirItem != '', 'Can not find the directory of the ground truth:'
        
        
        targetModalCubeDir = os.path.join(patientCubeDir, targetModalDirItem)
        makeDir(targetModalCubeDir)
        
        # targetModalCubeForSpecificSizeDir looks like './cubeData/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/7'
        targetModalCubeForSpecificSizeDir = os.path.join(targetModalCubeDir, str(cubeSize))
        makeDir(targetModalCubeForSpecificSizeDir)
        
        targetModalDir = os.path.join(patientDir, targetModalDirItem)
        goundTruthDir = os.path.join(patientDir, groundTruthDirItem)
        
        modalMhaFile = glob.glob(targetModalDir + '/*.mha')[0]
        groundTruthMhaFile = glob.glob(goundTruthDir + '/*.mha')[0]
        
        image3D = sitk.ReadImage(modalMhaFile)
        groundTruth = sitk.ReadImage(groundTruthMhaFile)
        
        image3DArray = sitk.GetArrayFromImage(image3D)
        groundTruthArray = sitk.GetArrayFromImage(groundTruth)
        
        groundTruthArrayCount = ndarrayCounter(groundTruthArray)
        groundTruthArrayCountDict = dict(groundTruthArrayCount)
        
        firstMax = groundTruthArrayCount.max()
        secondMax = groundTruthArrayCount[1:].max()
        
        negtivePointsAcceptRatio = ((secondMax + 200) / float(firstMax)) * 1.3
            
        stackVector2MatrixDic = dict(groundTruthArrayCount)
        cubeElementNumber = cubeSize * cubeSize * cubeSize
        for pair in groundTruthArrayCount:
            stackVector2MatrixDic[pair[0]] = []
        
#         print stackVector2MatrixDic
        groundTAShape = groundTruthArray.shape
        cubeHalfLen = (cubeSize - 1) / 2
        startIndex = (cubeHalfLen,) * 3
        
        for i in xrange(startIndex[0], groundTAShape[0] - cubeHalfLen):
            for j in xrange(startIndex[1], groundTAShape[1] - cubeHalfLen):
                for k in xrange(startIndex[2], groundTAShape[2] - cubeHalfLen):
                    # ijkLabel is the element in the groundTruth Array with index [i,j,k]
                    # ijkLabel represent the label of the voxel in the image3DArray with index [i,j,k]
                    ijkLabel = groundTruthArray[i, j, k]
                    if ijkLabel == 0 and random.random() > negtivePointsAcceptRatio:
                        continue
                    
                    ImageStartIndex = (i - cubeHalfLen, j - cubeHalfLen, k - cubeHalfLen)
                    ISIi, ISIj, ISIk = ImageStartIndex
                    cubeImage = image3DArray[ISIi: ISIi + cubeSize, ISIj: ISIj + cubeSize, ISIk: ISIk + cubeSize]
                    
                    vectorImage = cube2Vector(cubeImage)
                    vectorImageHead = [ijkLabel, i, j, k]
                    finalVectorImage = vectorImageHead + vectorImage
                    # TODO. The groundTruthArray has been cropped implictly, so there are something may be wrong
                    assert len(finalVectorImage) == len(vectorImage) + len(vectorImageHead) == cubeElementNumber + 4 
#                     print finalVectorImage
                    stackVector2MatrixDic[ijkLabel].append(finalVectorImage)
#             print stackVector2MatrixDic
        # stackedImageBaseName looks like '_cubeData_HGG_brats_2013_pat0001_1_VSD.Brain_3more.XX.O.OT.54517_7'
        stackedImageBaseName = targetModalCubeForSpecificSizeDir.replace('/', '_')[1:]
        # stackedImageName looks like '0_cubeData_HGG_brats_2013_pat0001_1_VSD.Brain_3more.XX.O.OT.54517_7'
        # Shuffle the negtive points
        random.shuffle(stackVector2MatrixDic[0])
        
        for key in stackVector2MatrixDic.keys():
            stackVector2MatrixLen = len(stackVector2MatrixDic[key])
            if key != 0:
                assert stackVector2MatrixLen == groundTruthArrayCountDict[key]
                assert stackVector2MatrixLen <= len(stackVector2MatrixDic[0])

            stackedImageArray = np.array(stackVector2MatrixDic[key])
            logger.debug('Shuffling the stackedImageArray, then it can be split in two part')
            np.random.shuffle(stackedImageArray)
            logger.debug('Shuffed the stackedImageArray, then it can be split in two part')
            numberOfValData = int(stackVector2MatrixLen * valDataRatio)
            numberOfTrainData = stackVector2MatrixLen - numberOfValData

            # The stackedTrainImageName looks like 0_54959_train_{cubeDirectory}_HGG_...
            stackedTrainImageName = str(key) + '_' + str(numberOfTrainData) + '_train' + stackedImageBaseName
            stackedTrainImageNameWithPath = os.path.join(targetModalCubeForSpecificSizeDir, stackedTrainImageName)
            stackedTrainImageArray = stackedImageArray[: numberOfTrainData]
            np.save(stackedTrainImageNameWithPath, stackedTrainImageArray)
            logger.debug(stackedTrainImageName +  'saved')

            stackedValImageName = str(key) + '_' + str(numberOfValData) + '_val' + stackedImageBaseName
            stackedValImageNameWithPath = os.path.join(targetModalCubeForSpecificSizeDir, stackedValImageName)
            stackedValImageArray = stackedImageArray[numberOfTrainData:]
            np.save(stackedValImageNameWithPath, stackedValImageArray)
            logger.debug(stackedValImageNameWithPath +  'saved')

    logger.info('The time for making cubes is {}'.format(time.time() - startTime))




def makeCubesOnTheFly(modal = 'T2', 
                      dataPath = '../data/BRATS2015_Training/', 
                      cubeDirectory = '../data/cubeData',
                      grade = 'HGG', 
        #               numberOfPatients = 10, 
        #               numberOfPointsPerPatient = 100, 
                      cubeSize = 5, 
                      valDataRatio = 0.1, 
                      parallel = False):
    '''
    According to the size of receptive field of network, generate the data points, i.e., the cubeSize should equal the size of receptive field
    Every generated data point have two part, the 3D data array with shape (cubeSize, cubeSize, cubeSize) and the its label
    The data label use a int number to represent, i.e., 1 for necrosis, 2 for edma, 3 for non-enhancing, 4 for enhancing, 0 for everything else
    -----------------
    subStructure: tumer substructure, can choose edema, solid, necrotic or non-enhancing. Note, not every patient contains all kind of substruce
    modal: a modal list. Choose different modal for different substructure,for example, T2 and FLAIR are usually used to segment edema
    dataPath: path to the BRATS data
    grade: choose different grade patient, such as HGG or LGG
    numberOfPatients: the number of patients that are used to generate data point
    numberOfPoints: the number of generated data points
    cubeSize: the shape of cube like 3D array, must be an odd number
    -----------------
    '''

    assert modal in ['T1', 'T1c', 'T2', 'FLAIR'], '''{} not in the list ['T1', 'T1c', 'T2', 'FLAIR']'''.format(modal)
    assert grade in ['HGG', 'LGG'], '''{} not in the list ['HGG', 'LGG']'''.format(grade)
    assert cubeSize % 2 == 1 and cubeSize > 0, '''{} is not a odd number or {} not more than 0'''.format(cubeSize, cubeSize)
    
    logger = logging.getLogger(__name__)

    if parallel:
        logger.info('Using multiprocessing')
        pool = mp.Pool()

    startTime = time.time()

    makeDir(cubeDirectory)
    gradeCubeDir = os.path.join(cubeDirectory, grade)
    makeDir(gradeCubeDir)
    # The cubeData directory tree is simular with the BRSTS2015_Training directory tree
    gradeDir = os.path.join(dataPath, grade)
    patientDirList = os.listdir(gradeDir)
    for patientDirItem in tqdm_notebook(patientDirList, desc = 'For every patient'):
        patientCubeDir = os.path.join(gradeCubeDir, patientDirItem)
        makeDir(patientCubeDir)
        
        patientDir = os.path.join(gradeDir, patientDirItem)
        modalDirList = os.listdir(patientDir)
        
        targetModalDirItem = ''
        groundTruthDirItem = ''
        for modalDirItem in modalDirList:
            modalDirItemSegment = modalDirItem.split('.')
            if 'MR_' + modal in modalDirItemSegment:
                targetModalDirItem = modalDirItem
            
            if 'OT' in modalDirItemSegment:
                groundTruthDirItem = modalDirItem
        
        assert targetModalDirItem != '', 'Can not find the directory of this modal:' + modal
        assert groundTruthDirItem != '', 'Can not find the directory of the ground truth:'
        
        
        targetModalCubeDir = os.path.join(patientCubeDir, targetModalDirItem)
        makeDir(targetModalCubeDir)
        
        # targetModalCubeForSpecificSizeDir looks like './cubeData/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/7'
        targetModalCubeForSpecificSizeDir = os.path.join(targetModalCubeDir, str(cubeSize))
        makeDir(targetModalCubeForSpecificSizeDir)
        
        targetModalDir = os.path.join(patientDir, targetModalDirItem)
        goundTruthDir = os.path.join(patientDir, groundTruthDirItem)
        
        modalMhaFile = glob.glob(targetModalDir + '/*.mha')[0]
        groundTruthMhaFile = glob.glob(goundTruthDir + '/*.mha')[0]
        
        # ---------------------------------------------------------------------------

        if parallel:

            pool.apply_async(makeCubeOnTheFly, args = (groundTruthMhaFile, 
                                                       targetModalCubeForSpecificSizeDir, 
                                                       cubeSize, 
                                                       valDataRatio))

        else:
            makeCubeOnTheFly(groundTruthMhaFile, targetModalCubeForSpecificSizeDir, cubeSize, valDataRatio)

    if parallel:
        pool.close()
        pool.join()


    logger.info('The time for making cubes is {}'.format(time.time() - startTime))



def makeCubeOnTheFly(groundTruthMhaFile, targetModalCubeForSpecificSizeDir, cubeSize, valDataRatio):

    logger = logging.getLogger(__name__)

    groundTruth = sitk.ReadImage(groundTruthMhaFile)
    groundTruthArray = sitk.GetArrayFromImage(groundTruth)

    groundTAShape = groundTruthArray.shape
    cubeHalfLen = (cubeSize - 1) / 2
    startIndex = (cubeHalfLen,) * 3

    effectiveGroundTruthArray = groundTruthArray[startIndex[0]:groundTAShape[0] - cubeHalfLen,
                                                 startIndex[1]:groundTAShape[1] - cubeHalfLen,
                                                 startIndex[2]:groundTAShape[2] - cubeHalfLen]
    
    groundTruthArrayCount = ndarrayCounter(groundTruthArray)
    groundTruthArrayCountDict = dict(groundTruthArrayCount)

    effectiveGroundTruthArrayCount = ndarrayCounter(effectiveGroundTruthArray)
    effectiveGroundTruthArrayCountDict = dict(effectiveGroundTruthArrayCount)

    logger.debug('groundTruthArrayCountDict: {}'.format(groundTruthArrayCountDict))

    for label in groundTruthArrayCountDict.keys():
        if label !=0 and groundTruthArrayCountDict[label] != effectiveGroundTruthArrayCountDict[label]:
            logger.warning('The number of label {} in ground truth array and \
                effective ground truth array \
                equal {}, {} respectively'.format(label,
                                                  len(groundTruthArrayCountDict[label]),
                                                  len(effectiveGroundTruthArrayCountDict[label])))
    
    firstMax = effectiveGroundTruthArrayCount.max()
    secondMax = effectiveGroundTruthArrayCount[1:].max()
    
    negtivePointsAcceptRatio = ((secondMax + 200) / float(firstMax)) * 3
        
    stackVector2MatrixDic = dict(effectiveGroundTruthArrayCount)
    cubeElementNumber = cubeSize * cubeSize * cubeSize
    for pair in effectiveGroundTruthArrayCount:
        stackVector2MatrixDic[pair[0]] = []
    
#         print stackVector2MatrixDic
    
    for i in xrange(startIndex[0], groundTAShape[0] - cubeHalfLen):
        for j in xrange(startIndex[1], groundTAShape[1] - cubeHalfLen):
            for k in xrange(startIndex[2], groundTAShape[2] - cubeHalfLen):
                # ijkLabel is the element in the groundTruth Array with index [i,j,k]
                # ijkLabel represent the label of the voxel in the image3DArray with index [i,j,k]
                ijkLabel = groundTruthArray[i, j, k]
                if ijkLabel == 0 and random.random() > negtivePointsAcceptRatio:
                    continue
                
                ImageStartIndex = (i - cubeHalfLen, j - cubeHalfLen, k - cubeHalfLen)
                ISIi, ISIj, ISIk = ImageStartIndex

                ImageEndIndex = (ISIi + cubeSize, ISIj + cubeSize, ISIk + cubeSize)
                IEIi, IEIj, IEIk = ImageEndIndex

                vectorImageHead = [ijkLabel, i, j, k, ISIi, IEIi, ISIj, IEIj, ISIk, IEIk]

                stackVector2MatrixDic[ijkLabel].append(vectorImageHead)

    stackedImageBaseName = targetModalCubeForSpecificSizeDir.replace('/', '_')[2:]
    logger.debug('stackedImageBaseName: {}'.format(stackedImageBaseName))
    # stackedImageName looks like '0_cubeData_HGG_brats_2013_pat0001_1_VSD.Brain_3more.XX.O.OT.54517_7'
    # Shuffle the negtive points
    random.shuffle(stackVector2MatrixDic[0])
    
    for key in stackVector2MatrixDic.keys():
        stackVector2MatrixLen = len(stackVector2MatrixDic[key])
        if key != 0:
            assert stackVector2MatrixLen == effectiveGroundTruthArrayCountDict[key]
            assert stackVector2MatrixLen <= len(stackVector2MatrixDic[0]), '{}, {}'.format(stackVector2MatrixLen, 
                                                                                           len(stackVector2MatrixDic[0]))

        stackedImageArray = np.array(stackVector2MatrixDic[key])
        logger.debug('Shuffling the stackedImageArray, then it can be split in two part')
        np.random.shuffle(stackedImageArray)
        logger.debug('Shuffed the stackedImageArray, then it can be split in two part')
        numberOfValData = int(stackVector2MatrixLen * valDataRatio)
        numberOfTrainData = stackVector2MatrixLen - numberOfValData

        # The stackedTrainImageName looks like 0_54959_train_{cubeDirectory}_HGG_...
        stackedTrainImageName = str(key) + '_' + str(numberOfTrainData) + '_train' + stackedImageBaseName
        stackedTrainImageNameWithPath = os.path.join(targetModalCubeForSpecificSizeDir, stackedTrainImageName)
        stackedTrainImageArray = stackedImageArray[: numberOfTrainData]
        np.save(stackedTrainImageNameWithPath, stackedTrainImageArray)
        logger.debug(stackedTrainImageName +  'saved')

        stackedValImageName = str(key) + '_' + str(numberOfValData) + '_val' + stackedImageBaseName
        stackedValImageNameWithPath = os.path.join(targetModalCubeForSpecificSizeDir, stackedValImageName)
        stackedValImageArray = stackedImageArray[numberOfTrainData:]
        np.save(stackedValImageNameWithPath, stackedValImageArray)
        logger.debug(stackedValImageNameWithPath +  'saved')



class cubesGetor():
    
    def __init__(self, 
                 subStructure = 'edema', 
                 modal = 'T2', 
                 batchSize = 20,
                 cubeDirectory = '../data/cubeData',
                 sourceDataDirectory = '../data/normalizedDataSet',
                 grade = 'HGG', 
                 cubeSize = 5, 
                 negPosRatio = 1., 
                 trainOrVal = 'train',
                 onTheFly = True):
        self.subStructure = subStructure
        self.modal = modal
        self.batchSize = batchSize
        self.cubeDirectory = cubeDirectory
        self.sourceDataDirectory = sourceDataDirectory
        self.grade = grade
        self.cubeSize = cubeSize
        self.negPosRatio = negPosRatio
        self.trainOrVal = trainOrVal
        self.onTheFly = onTheFly

        self.logger = logging.getLogger(__name__)
        
        # gradeCubeDir looks like {self.cubeDirectory}/HGG
        gradeCubeDir = os.path.join(self.cubeDirectory, self.grade)

        patientCubeDirItemList = os.listdir(gradeCubeDir)
        # patientCubeDirItem looks like brats_2013_pat0001_1

        patientCubeDirList = [os.path.join(gradeCubeDir, patientCubeDirItem) \
                          for patientCubeDirItem in patientCubeDirItemList]

        self.patientCubeDirList = patientCubeDirList


        self.logger.debug('Found {} {} patients'.format(len(self.patientCubeDirList), self.grade))
        
        dataInformationList = self.goToDataDirectly()
        self.logger.debug('dataInformationList: {}'.format(dataInformationList))
        # sourceDataInformationList = self.goToDataDirectly(toSource = True)


        self.dataFileList = [fileInfoItem[0] for fileInfoItem in dataInformationList]
        self.dataFileLengthList = [fileInfoItem[1] for fileInfoItem in dataInformationList]
        self.dataFileLengthAddedList = [sum(self.dataFileLengthList[: i + 1]) \
                                        for i in xrange(len(self.dataFileLengthList))]

        self.sourceDataFileList = []

        for dataFileItem in self.dataFileList:
            dataFileItemSegment = dataFileItem.split('/')
            neededDataFileItemSeg = dataFileItemSegment[3:6]

            sourceDataDir = os.path.join(self.sourceDataDirectory, *neededDataFileItemSeg)
            sourceDataFileItem = glob.glob(sourceDataDir + '/*.mha')[0]

            self.sourceDataFileList.append(sourceDataFileItem)

        self.logger.debug('sourceDataFileList: {}'.format(self.sourceDataFileList))


    def goToDataDirectly(self):
        subStructureDict = {'other': 0, 
                            'necrosis': 1,
                            'edema': 2,
                            'non-enhancing': 3,
                            'enhancing': 4}
        dataInfomationList = []

        for patientCubeDir in self.patientCubeDirList:
            
            modalCubeDirList = os.listdir(patientCubeDir)
            targetModalCubeDirItem = ''
            
            # modalCubeDirItem looks like VSD.Brain.XX.O.MR_T2.54515
            for modalCubeDirItem in modalCubeDirList:
                modalCubeDirItemSegment = modalCubeDirItem.split('.')
                if 'MR_' + self.modal in modalCubeDirItemSegment:
                    targetModalCubeDirItem = modalCubeDirItem

            # targetModalCubeDir looks like {self.cubeDirectory}/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T2.54515
            targetModalCubeDir = os.path.join(patientCubeDir, targetModalCubeDirItem)
            cubeDataFileDir = os.path.join(targetModalCubeDir, str(self.cubeSize))
            cubeDataFileDirList = os.listdir(cubeDataFileDir)
            cubeDataFileDirList.sort(key = lambda fileName: fileName[0])
            
            targetFileItem = ''
            negtiveFileItem = ''

            for cubeDataFileDirItem in cubeDataFileDirList:
                cubeDataFileDirItemSegment = cubeDataFileDirItem.split('_')
                self.logger.debug('trainOrVal: {}, file infomation: {}'.format(self.trainOrVal, 
                                                                         cubeDataFileDirItemSegment[2]))

                if self.trainOrVal == cubeDataFileDirItemSegment[2]:

                    if str(subStructureDict[self.subStructure]) == cubeDataFileDirItemSegment[0]:
                        targetFileItem = cubeDataFileDirItem

                    if str(subStructureDict['other']) == cubeDataFileDirItemSegment[0]:
                        negtiveFileItem = cubeDataFileDirItem

            assert targetFileItem != '' and negtiveFileItem != ''
            
            targetFile = os.path.join(cubeDataFileDir, targetFileItem)
            negtiveFile = os.path.join(cubeDataFileDir, negtiveFileItem)

            self.logger.debug('Find the {} positive file: {}'.format(self.trainOrVal, targetFile))
            self.logger.debug('Find the {} negtive file: {}'.format(self.trainOrVal, negtiveFile))
            
            # The number of negtive points usually large that the positive's
            numberOfDataItems = targetFileItem.split('_')[1]

            numberOfNegtiveDataItems = int(self.negPosRatio * int(numberOfDataItems))

            assert numberOfNegtiveDataItems <= int(negtiveFileItem.split('_')[1])
            
            dataInfomationList.append((targetFile, int(numberOfDataItems)))
            dataInfomationList.append((negtiveFile, numberOfNegtiveDataItems))
            
        return dataInfomationList

        
    def fetchCubes(self, batchSize = 0, shuffle = True, oneHot = False, parallel = False):
        dataIndexList = range(1, self.dataFileLengthAddedList[-1] + 1)

        if batchSize == 0:
            batchSize = self.batchSize
        
        assert batchSize <= len(dataIndexList)

        self.logger.info('The number of data points is {}'.format(len(dataIndexList)))
        assert len(dataIndexList) == self.dataFileLengthAddedList[-1]
        if shuffle:
            self.logger.info('Shuffling the dataIndexList')
            random.shuffle(dataIndexList)
        
        for startIndex in range(0, len(dataIndexList) - batchSize + 1, batchSize):
            self.logger.debug('Fetching the {} th of {} excepts in {} data'.format(startIndex / batchSize + 1, len(dataIndexList) /batchSize, self.trainOrVal))
            excerpt = dataIndexList[startIndex: startIndex + batchSize]
            
            if parallel:

                results = []
                pool = pmp.Pool()

                for dataIndex in excerpt:

                    result = pool.apply_async(self.fromIdxGetDataItem, args = (dataIndex,))
                    results.append(result)

                pool.close()
                pool.join()


                batchDataItemList = [res.get() for res in results]

                assert len(batchDataItemList) == batchSize, 'batchDataItemList: {}, batchSize: {}'.format(batchDataItemList,
                                                                                                          batchSize)

            else:
                batchDataItemList = [self.fromIdxGetDataItem(dataIndex) for dataIndex in excerpt]

            batchDataLabelList = [batchDataItemList[i][0] for i in range(batchSize)]
            batchDataCubeList = [batchDataItemList[i][1] for i in range(batchSize)]
            
            batchDataLabelOneHot = []
            
            
            if oneHot:

                for label in batchDataLabelList:
                   
                    if label == 0:
                        batchDataLabelOneHot.append([1, 0])
                    elif label > 0:
                        batchDataLabelOneHot.append([0, 1])
                    else:
                        raise

                reshapedBatchData = np.array(batchDataCubeList).reshape(
                    (batchSize, 1, self.cubeSize, self.cubeSize, self.cubeSize))
                reshapedBatchLabel = np.array(batchDataLabelOneHot).reshape(
                    (batchSize, 2, 1, 1, 1))
                
            else:
                reshapedBatchData = np.array(batchDataCubeList, dtype = 'float32').reshape(
                    (batchSize, 1, self.cubeSize, self.cubeSize, self.cubeSize))

                reshapedBatchLabel = np.array(batchDataLabelList, dtype = 'int32')

            yield reshapedBatchData, reshapedBatchLabel

  
    def fromIdxGetDataItem(self, dataIndex):
        assert dataIndex >= 1 and dataIndex <= self.dataFileLengthAddedList[-1]
        fileIndex = bisect.bisect_left(self.dataFileLengthAddedList, dataIndex)
        assert dataIndex <= self.dataFileLengthAddedList[fileIndex]

        if fileIndex == 0:
            relativeIndex = dataIndex - 1
        if fileIndex > 0:
            assert dataIndex > self.dataFileLengthAddedList[fileIndex - 1]
            relativeIndex = dataIndex - self.dataFileLengthAddedList[fileIndex - 1] - 1
        
        stackedImageArray = np.load(self.dataFileList[fileIndex])
        dataItem = stackedImageArray[relativeIndex]
        dataLabel = dataItem[0]
        dataOriginalIndex = dataItem[1: 4]

        if not self.onTheFly:
            dataVector = dataItem[4:]
            dataCube = vector2Cube(dataVector, self.cubeSize)

        if self.onTheFly:
            ISIi, IEIi, ISIj, IEIj, ISIk, IEIk = dataItem[4:]
            image3D = sitk.ReadImage(self.sourceDataFileList[fileIndex])
            image3DArray = sitk.GetArrayFromImage(image3D)
            dataCube = image3DArray[ISIi: IEIi, ISIj: IEIj, ISIk: IEIk]
        
        return (dataLabel, dataCube)

    
    def generateSynthesisData(self, batchSize = 0, dataSize = 100000, group0 = (0, 1), group1 = (1, 1)):

        if batchSize == 0:
            batchSize = self.batchSize

        batchNumber = dataSize / batchSize

        for i in xrange(batchNumber):
            batchData = []
            batchLabel = []

            for j in xrange(batchSize):
                negtiveRatio = self.negPosRatio / (1. + self.negPosRatio)

                if np.random.random() < negtiveRatio:
                    batchData.append(np.random.normal(*group0, size = (self.cubeSize,) * 3))
                    batchLabel.append(0)
                else:
                    batchData.append(np.random.normal(*group1, size = (self.cubeSize,) * 3))
                    batchLabel.append(1)

            reshapedBatchData = np.array(batchData, dtype = theano.config.floatX).reshape(
                (batchSize, 1, self.cubeSize, self.cubeSize, self.cubeSize))

            reshapedBatchLabel = np.array(batchLabel, dtype = 'int32')

            yield reshapedBatchData, reshapedBatchLabel


# def _pickle_method(m):
#     if m.im_self is None:
#         return getattr, (m.im_class, m.im_func.func_name)
#     else:
#         return getattr, (m.im_self, m.im_func.func_name)

# copy_reg.pickle(types.MethodType, _pickle_method)


