"""EWC hook"""
import chainer
import numpy
import six
from chainer import cuda


class EWC(object):
    """Optimizer hook function for elastic weight consolidation regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'ElasticWeightConsolidation'

    def __init__(self, rate, target, batchsize=1, device=-1):
        self.rate = rate
        self.target = target
        self.batchsize = batchsize
        self.device = device
        self.converter = chainer.dataset.convert.concat_examples

    def setup(self, lossfun, *args, **kwargs):
        self._states = {}
        self.prepare()
        x = args[0]
        self.origin_data_size = len(x)
        # --- Init Fisher matrix & param_origin ---
        states = self._states
        for name, param in self.target.namedparams():
            with cuda.get_device(param.data):
                state = states[name]
                state['param_origin'] = param.data.copy()
                xp = cuda.get_array_module(param.data)
                state['F'] = xp.zeros_like(param.data)

        for i in range(0, self.origin_data_size, self.batchsize):
            #minibatch_x = args[0][[i]][0]
            #minibatch_y = args[0][[i]][1]

            in_arrays = self.converter(args[0][i:i+self.batchsize], self.device)
            losses = lossfun(*in_arrays, reduce='no')  # lossfun(*args, **kwds)
            #minibatch_x = minibatch_x.reshape(1, -1)
            #minibatch_y = minibatch_y.reshape(1, -1)

            #loss = lossfun(minibatch_x, minibatch_y) # lossfun(*args, **kwds)
            # print('losses.shape', losses.shape)     # (batchsize,)
            for j, loss in enumerate(losses):
                iter = i + j
                self.target.cleargrads()
                loss.backward()
                del loss
                #self.target(minibatch_x, minibatch_y)
                for name, param in self.target.namedparams():
                    p, g = param.data, param.grad
                    #print('in_arrays0.shape', in_arrays[0].shape,
                    #      'in_arrays1.shape', in_arrays[1].shape,
                    #      'g.shape', g.shape)
                    with cuda.get_device(param.data) as dev:
                        state = states[name]
                        if int(dev) == -1:
                            state['F'] += g * g / self.origin_data_size
                        else:
                            # TODO: Review. This seems slow...
                            self.kernel_calc_fisher()(g, self.origin_data_size, state['F'])


    def prepare(self):
        """Prepares for an update.

        This method initializes missing optimizer states (e.g. for newly added
        parameters after the set up), and copies arrays in each state
        dictionary to CPU or GPU according to the corresponding parameter
        array.

        """
        states = self._states
        for name, param in self.target.namedparams():
            if name not in states:
                state = {}
                self.init_state(param, state)
                states[name] = state
            else:
                state = states[name]
                with cuda.get_device(param.data) as dev:
                    if int(dev) == -1:  # cpu
                        for key, value in six.iteritems(state):
                            if isinstance(value, cuda.ndarray):
                                state[key] = value.get()
                    else:  # gpu
                        cupy = cuda.cupy
                        for key, value in six.iteritems(state):
                            if isinstance(value, numpy.ndarray):
                                state[key] = cuda.to_gpu(value)
                            elif (isinstance(value, cupy.ndarray) and
                                  value.device != dev):
                                state[key] = cupy.copy(value)

    def init_state(self, param, state):
        """Initializes the optimizer state corresponding to the parameter.

        This method should add needed items to the ``state`` dictionary. Each
        optimizer implementation that uses its own states should override this
        method or CPU/GPU dedicated versions (:meth:`init_state_cpu` and
        :meth:`init_state_gpu`).

        Args:
            param (~chainer.Variable): Parameter variable.
            state (dict): State dictionary.

        .. seealso:: :meth:`init_state_cpu`, :meth:`init_state_gpu`

        """
        pass
        #with cuda.get_device(param.data) as dev:
            #if int(dev) == -1:
                #self.init_state_cpu(param, state)
            #else:
                #self.init_state_gpu(param, state)

    def kernel(self):
        return cuda.elementwise(
            'T p, T decay, T F, T param_origin', 'T g',
            'g += decay * F * (p - param_origin)',
            'elastic_weight_consolidation')

    def kernel_calc_fisher(self):
        return cuda.elementwise(
            'T g, T data_size', 'T F',
            'F += g * g / data_size',
            'calc_fisher')

    def __call__(self, opt):
        rate = self.rate
        states = self._states
        for name, param in self.target.namedparams():
            p, g = param.data, param.grad
            state = states[name]
            with cuda.get_device(param.data) as dev:
                if int(dev) == -1:
                    g += rate * state['F'] * (p - state['param_origin'])
                else:
                    self.kernel()(p, rate, state['F'], state['param_origin'], g)
