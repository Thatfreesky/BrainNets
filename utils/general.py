'''
Some short simple and helper function at here.
'''

import os
import logging
import shutil

def makeDir(dir, force = False):

    '''
    The force parameter is dangerous. Be careful to use it.
    '''

    logger = logging.getLogger(__name__)

    if os.path.exists(dir):
        logger.debug('\nThe dir, {}, has already been here.'.format(dir))

        if force:
            shutil.rmtree(dir)
            logger.debug('\nForce to remove the exist dir, {}'.format(dir))

            os.mkdir(dir)
            logger.debug('\nHave made the new dir, {}, after removing the old one.'.format(dir))

        else:
            logger.debug('\nFailed to create the dir, {}, \
                         because there exist one and force == False'.format(dir))

    else:
        os.mkdir(dir)
        logger.debug('\nSucessfully created the dir, {}'.format(dir))


def numpyArrayCounter(numpyArray):

    unique, counts = np.unique(numpyArray, return_counts = True)

    return np.asarray((unique, counts)).T



def logMessage(sample, message):

    # So, the message should better not long than 80

    messageLen = len(message)

    maxLen = 130
    maxLenStr = '{}'.format(maxLen)

    return '\n' + '{:{padSampl}{align}{width}}'.format(message, padSampl = sample, align = '^', width = maxLenStr)
    


def logTable(tableRowList):

    maxLen = 130

    columnNum = len(tableRowList[1])

    verticalLineNum = columnNum - 1
    columnLen = (maxLen - verticalLineNum) / columnNum
    remainLen = maxLen - columnLen * columnNum - verticalLineNum

    columnLenList = list((columnLen,) * columnNum)

    for idx in xrange(remainLen):
        columnLenList[idx] += 1

    assert sum(columnLenList) + verticalLineNum == maxLen

    pattern = '{:^{length}}'
    table = ''

    for i, row in enumerate(tableRowList):
        for j, column in enumerate(row):
            table += pattern.format(column, length = columnLenList[j])
            if j < columnNum - 1:
                table += '|'
            else:
                table += '\n'

    table = '\n' + table

    return table
