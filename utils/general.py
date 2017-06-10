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
        logger.debug('The dir, {}, has already been here.'.format(dir))

        if force:
            shutil.rmtree(dir)
            logger.debug('Force to remove the exist dir, {}'.format(dir))

            os.mkdir(dir)
            logger.debug('Have made the new dir, {}, after removing the old one.'.format(dir))

        else:
            logger.debug('Failed to create the dir, {}, \
                         because there exist one and force == False'.format(dir))

    else:
        os.mkdir(dir)
        logger.debug('Sucessfully created the dir, {}'.format(dir))


def numpyArrayCounter(numpyArray):

    unique, counts = np.unique(numpyArray, return_counts = True)

    return np.asarray((unique, counts)).T


