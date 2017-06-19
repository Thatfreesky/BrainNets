import theano.tensor as T
from lasagne import init, nonlinearities
from lasagne.utils import as_tuple
from lasagne.layers.base import Layer
from lasagne.layers.conv import BaseConvLayer, conv_input_length



class DilatedConv3DLayer(BaseConvLayer):
    """
    lasagne.layers.DilatedConv3DLayer(incoming, num_filters, filter_size,
    dilation=(1, 1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs)

    3D dilated convolution layer

    Performs a 3D convolution with dilated filters, then optionally adds a bias
    and applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape
        ``(batch_size, num_input_channels, input_depth, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    dilation : int or iterable of int
        An integer or a 3-element tuple specifying the dilation factor of the
        filters. A factor of :math:`x` corresponds to :math:`x - 1` zeros
        inserted between adjacent filter elements.

    pad : int, iterable of int, or 'valid' (default: 0)
        The amount of implicit zero padding of the input.
        This implementation does not support padding, the argument is provided
        for compatibility to other convolutional layers only.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape
        ``(num_input_channels, num_filters, input_depth, filter_rows, filter_columns)``.
        Note that the first two dimensions are swapped compared to a
        non-dilated convolution.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False)
        Whether to flip the filters before sliding them over the input,
        performing a convolution, or not to flip them and perform a
        correlation (this is the default).
        This implementation does not support flipped filters, the argument is
        provided for compatibility to other convolutional layers only.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Notes
    -----
    The dilated convolution is implemented as the backward pass of a
    convolution wrt. weights, passing the filters as the output gradient.
    It can be thought of as dilating the filters (by adding ``dilation - 1``
    zeros between adjacent filter elements) and cross-correlating them with the
    input. See [1]_ for more background.

    References
    ----------
    .. [1] Fisher Yu, Vladlen Koltun (2016),
           Multi-Scale Context Aggregation by Dilated Convolutions. ICLR 2016.
           http://arxiv.org/abs/1511.07122, https://github.com/fyu/dilation
    """
    def __init__(self, incoming, num_filters, filter_size, dilation=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 **kwargs):
        self.dilation = as_tuple(dilation, 3, int)
        super(DilatedConv3DLayer, self).__init__(
                incoming, num_filters, filter_size, 1, pad,
                untie_biases, W, b, nonlinearity, flip_filters, n=3, **kwargs)
        # remove self.stride:
        del self.stride
        # require valid convolution
        if self.pad != (0, 0, 0):
            raise NotImplementedError(
                    "DilatedConv2DLayer requires pad=0 / (0,0) / 'valid', but "
                    "got %r. For a padded dilated convolution, add a PadLayer."
                    % (pad,))
        # require unflipped filters
        if self.flip_filters:
            raise NotImplementedError(
                    "DilatedConv2DLayer requires flip_filters=False.")

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, (filter-1) * dilate + 1, 1, 0)
                      for input, filter, dilate
                      in zip(input_shape[2:], self.filter_size,
                             self.dilation)))

    def convolve(self, input, **kwargs):
        # we perform a convolution backward pass wrt weights,
        # passing kernels as output gradient
        imshp = self.input_shape
        kshp = self.output_shape
        # and swapping channels and batchsize
        imshp = (imshp[1], imshp[0]) + imshp[2:]
        kshp = (kshp[1], kshp[0]) + kshp[2:]
        op = T.nnet.abstract_conv.AbstractConv3d_gradWeights(
            imshp=imshp, kshp=kshp,
            subsample=self.dilation, border_mode='valid',
            filter_flip=False)
        output_size = self.output_shape[2:]
        if any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(input.transpose(1, 0, 2, 3, 4), self.W, output_size)
        return conved.transpose(1, 0, 2, 3, 4)
