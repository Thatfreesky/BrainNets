import nibabel as nib
import numpy as np
import os
import logging
import shutil
import theano
import pickle
from nipype.interfaces.ants import N4BiasFieldCorrection
from medpy.filter import IntensityRangeStandardization


import general as ge



def preProcessingWithN4(inputDir,
                        modals = ['t1ce', 't1'],
                        bsplineFittingDistance = 200,
                        shrinkFactor = 2,
                        iterations = [20,20,20,10]):

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

    for patientDir, modalFileName in ge.goToTheImageFiles(inputDir):

        modalFileNameSegList = modalFileName.split('_')
        modalNameWithFileType = modalFileNameSegList[-1]

        modalName = modalNameWithFileType.split('.')[0]

        assert modalName in ['flair', 't2', 't1ce', 't1', 'seg']

        if modalName in modals:

            inputImagePath = os.path.join(patientDir, modalFileName)
            outputImagePath = os.path.join(patientDir, 'N4' + modalFileName)

            N4BiasCorrectAFile(inputImagePath,
                               bsplineFittingDistance,
                               shrinkFactor,
                               iterations,
                               outputImagePath)


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


def IntensityRangesStand():

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


def trainIntensityRangeStandModel(imageDir,
                                  grade,
                                  modal,
                                  afterN4,
                                  numUsedToTrain,
                                  storeLocation,
                                  cutoff = (1,99),
                                  landmark = 'L4',
                                  stdrange = 'auto'):

    logger = logging.getLogger(__name__)

    for patientDir, modalFileName in ge.goToTheImageFiles(inputDir, grade):

        if afterN4 and 'N4' not in modalFileName: continue
        if not afterN4 and 'N4' in modalFileName: continue

        modalFileNameSegList = modalFileName.split('_')
        modalNameWithFileType = modalFileNameSegList[-1]

        modalName = modalNameWithFileType.split('.')[0]

        assert modalName in ['flair', 't2', 't1ce', 't1', 'seg']

        if modalName in modals:





    
















    

    