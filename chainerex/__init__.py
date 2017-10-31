from chainerex import dataset  # NOQA
from chainerex import iterators  # NOQA
from chainerex import links  # NOQA
from chainerex import optimizers  # NOQA
from chainerex import training  # NOQA
from chainerex import utils  # NOQA

import logging
DEBUG = True
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)


dataset.indexers.feature_indexer.install_features_indexer()
