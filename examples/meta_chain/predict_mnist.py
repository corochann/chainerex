import numpy as np
from chainer.datasets import get_mnist

from examples.meta_chain.train_mnist import MLP

train, test = get_mnist()

save_dir = 'store_model'
predictor = MLP.load(save_dir)
x, y, t = predictor.predict(test[:20])

#print('y', y)
print('y predict', np.argmax(y, axis=1))
print('t', t)
