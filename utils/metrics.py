import numpy as np
import logging


def voxleWiseMetrics(npArray1, npArray2, labelList):

    '''
    npArray1 is the prediction array.
    npArray2 is the target array.
    '''

    logger = logging.getLogger(__name__)

    pArray = np.zeros(npArray1.shape, dtype = npArray1.dtype)
    tArray = np.zeros(npArray2.shape, dtype = npArray2.dtype)

    for label in labelList:
        pArray += (npArray1 == label).astype(int)
        tArray += (npArray2 == label).astype(int)

    assert np.sum(pArray) != 0
    assert np.sum(tArray) != 0

    p1Array = (pArray != 0).astype(int)
    t1Array = (tArray != 0).astype(int)

    p1ArrayNum = np.sum(p1Array)
    t1ArrayNum = np.sum(t1Array)

    p1Andt1Array = (p1Array * t1Array).astype(int)
    p1Andt1ArrayNum = np.sum(p1Andt1Array)
    assert p1Andt1ArrayNum <= p1ArrayNum, \
          'p1Andt1ArrayNum: {}, p1ArrayNum: {}'.format(p1Andt1ArrayNum, p1ArrayNum)
    assert p1Andt1ArrayNum <= t1ArrayNum, \
          'p1Andt1ArrayNum: {}, t1ArrayNum: {}'.format(p1Andt1ArrayNum, t1ArrayNum)

    p0Array = (pArray == 0).astype(int)
    t0Array = (tArray == 0).astype(int)

    p0ArrayNum = np.sum(p0Array)
    t0ArrayNum = np.sum(t0Array)

    p0Andt0Array = (p0Array * t0Array).astype(int)
    p0Andt0ArrayNum = np.sum(p0Andt0Array)
    assert p0Andt0ArrayNum <= p0ArrayNum, \
          'p0Andt0ArrayNum: {}, p0ArrayNum: {}'.format(p0Andt0ArrayNum, p0ArrayNum)
    assert p0Andt0ArrayNum <= t0ArrayNum, \
          'p0Andt0ArrayNum: {}, t0ArrayNum: {}'.format(p0Andt0ArrayNum, t0ArrayNum)

    assert p1ArrayNum + p0ArrayNum == pArray.size
    assert t1ArrayNum + t0ArrayNum == tArray.size
    

    diceScore = 2.0 * p1Andt1ArrayNum / (p1ArrayNum + t1ArrayNum)

    sensitivity = p1Andt1ArrayNum / float(t1ArrayNum)

    specificity = p0Andt0ArrayNum / float(t0ArrayNum)

    return diceScore, sensitivity, specificity