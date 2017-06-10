import numpy as np
import theano
import logging
import theano.tensor as T
from lasagne.init impoet HeUniform
from lasagne.updates import momentum
from lasagne.layers import (InputLayer,
                            ConcatLayer,
                            Convo3DLayer,
                            Pool3DLayer,
                            batch_norm,
                            NonlinearityLayer,
                            get_all_params, 
                            DimshuffleLayer,
                            get_output, 
                            set_all_param_values, 
                            get_output_shape, 
                            ReshapeLayer)

from lasagne.nonlinearities import rectify, linear, softmax
from lasagne.objectives import categorical_crossentropy

class BaseNet():

    logger = logging.getLogger(__name__)

    def __init__(self, modalConfigFile)