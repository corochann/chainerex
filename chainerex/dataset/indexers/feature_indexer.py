import six
import numpy
from logging import getLogger

from chainer.dataset.dataset_mixin import DatasetMixin
from chainer.datasets.concatenated_dataset import ConcatenatedDataset
from chainer.datasets.dict_dataset import DictDataset
from chainer.datasets.image_dataset import ImageDataset
from chainer.datasets.image_dataset import LabeledImageDataset
from chainer.datasets.sub_dataset import SubDataset
from chainer.datasets.transform_dataset import TransformDataset
from chainer.datasets.tuple_dataset import TupleDataset

from chainerex.dataset.indexers.indexer import BaseIndexer, ExtractBySliceNotSupportedError  # NOQA


class BaseFeatureIndexer(BaseIndexer):

    """Base class for FeatureIndexer

    FeatureIndexer can be accessed by 2-dimensional indices, axis=0 is used for
    dataset index and axis=1 is used for feature index.
    For example, let `features` be the instance of `BaseFeatureIndexer`, then
    `features[i, j]` returns `i`-th dataset of `j`-th feature.

    `features[ind]` works same with `features[ind, :]`

    Note that the returned value will be numpy array, even though the
    dataset is initilized with other format (e.g. list).

    """

    def __init__(self, dataset):
        super(BaseFeatureIndexer, self).__init__()
        self.dataset = dataset

    def features_length(self):
        """Returns length of features

        Returns (int): feature length

        """
        raise NotImplementedError

    def dataset_length(self):
        return len(self.dataset)

    @property
    def shape(self):
        return self.dataset_length(), self.features_length()

    def extract_feature_by_slice(self, slice_index, j):
        """Extracts `slice_index`-th data's `j`-th feature.

        Here, `slice_index` is indices of slice object.
        This method may be override to support efficient feature extraction.
        If not override, `ExtractBySliceNotSupportedError` is raised by
        default, and in this case `extract_feature` is used instead.

        Args:
            slice_index (slice): slice of data index to be extracted
            j (int): `j`-th feature to be extracted

        Returns: feature
        """

        raise ExtractBySliceNotSupportedError

    def extract_feature(self, i, j):
        """Extracts `i`-th data's `j`-th feature

        Args:
            i (int): `i`-th data to be extracted
            j (int): `j`-th feature to be extracted

        Returns: feature

        """
        raise NotImplementedError

    def create_feature_index_list(self, feature_index):
        if isinstance(feature_index, slice):
            feature_index_list = numpy.arange(
                *feature_index.indices(self.features_length())
            )
        elif isinstance(feature_index, (list, numpy.ndarray)):
            if isinstance(feature_index[0],
                          (bool, numpy.bool, numpy.bool_)):
                if len(feature_index) != self.features_length():
                    raise ValueError('Feature index wrong length {} instead of'
                                     ' {}'.format(len(feature_index),
                                                  self.features_length()))
                feature_index_list = numpy.argwhere(feature_index
                                                    ).ravel()
            else:
                feature_index_list = feature_index
        else:
            # assuming int type
            feature_index_list = [feature_index]
        # feature_index_list may contain negative value index, so convert them.
        feature_index_list = [self.features_length() + i if i < 0 else i
                              for i in feature_index_list]
        return feature_index_list

    def preprocess(self, item):
        pass

    def postprocess(self, item):
        pass

    def __getitem__(self, item):
        self.preprocess(item)
        if isinstance(item, tuple):
            index_dim = len(item)
            # multi dimensional access
            if index_dim == 1:
                # This is not unexpected case...
                data_index = item[0]
                feature_index_list = self.create_feature_index_list(
                    slice(None)
                )
            elif index_dim == 2:
                data_index, feature_index = item
                feature_index_list = self.create_feature_index_list(
                    feature_index
                )
            else:
                raise IndexError('too many indices for features')
        else:
            data_index = item
            feature_index_list = self.create_feature_index_list(slice(None))
        if len(feature_index_list) == 1:
            self._extract_single_feature = True
            ret = self._extract_feature(data_index, feature_index_list[0])
        else:
            self._extract_single_feature = False
            ret = tuple([self._extract_feature(data_index, j) for j in
                         feature_index_list])
        self.postprocess(item)
        return ret

    def check_type_feature_index(self, j):
        if j >= self.features_length():
            raise IndexError('index {} is out of bounds for axis 1 with '
                             'size {}'.format(j, self.features_length()))

    def _extract_feature(self, data_index, j):
        """Format `data_index` and call proper method to extract feature.

        Args:
            data_index (int, slice, list or numpy.ndarray):
            j (int or key):

        """
        self.check_type_feature_index(j)
        if isinstance(data_index, slice):
            try:
                return self.extract_feature_by_slice(data_index, j)
            except ExtractBySliceNotSupportedError:
                # Accessing by each index, copy occurs
                current, stop, step = data_index.indices(self.dataset_length())
                res = [self.extract_feature(i, j) for i in
                       six.moves.range(current, stop, step)]
        elif isinstance(data_index, (list, numpy.ndarray)):
            if isinstance(data_index[0], (bool, numpy.bool, numpy.bool_)):
                # Access by bool flag list
                if len(data_index) != self.dataset_length():
                    raise ValueError('Feature index wrong length {} instead of'
                                     ' {}'.format(len(data_index),
                                                  self.dataset_length()))
                data_index = numpy.argwhere(data_index).ravel()

            # assuming data_index is list.
            # it may contain negative value index, so convert them.
            data_index = [self.dataset_length() + i if i < 0 else i
                          for i in data_index]
            if len(data_index) == 1:
                return self.extract_feature(data_index[0], j)
            else:
                res = [self.extract_feature(i, j) for i in data_index]
        else:
            # assuming data_index is int.
            # it may contain negative value index, so convert them.
            if data_index < 0:
                data_index += self.dataset_length()
            return self.extract_feature(data_index, j)
        try:
            feature = numpy.asarray(res)
        except ValueError:
            feature = numpy.empty(len(res), dtype=object)
            feature[:] = res[:]
        return feature


class TupleDatasetEx(object):
    _features_indexer = None

    @property
    def td_features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index

        .. admonition:: Example

           >>> from chainer.datasets import TupleDataset
           >>> tuple_dataset = TupleDataset([0, 1, 2], [0, 1, 4])
           >>> targets = tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        if self._features_indexer is None:
            self._features_indexer = TupleDatasetFeatureIndexer(self)
        return self._features_indexer


class TupleDatasetFeatureIndexer(BaseFeatureIndexer):
    """FeatureIndexer for TupleDataset"""

    def __init__(self, dataset):
        """
        
        Args:
            dataset (TupleDataset): 
        """
        if not isinstance(dataset, TupleDataset):
            raise TypeError('dataset class {} is not expected'
                            .format(type(dataset)))
        super(TupleDatasetFeatureIndexer, self).__init__(dataset)
        self.datasets = dataset._datasets

    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]


@property
def td_features(self):
    """Extract features according to the specified index.

    - axis 0 is used to specify dataset id (`i`-th dataset)
    - axis 1 is used to specify feature index

    .. admonition:: Example

       >>> from chainer.datasets import TupleDataset
       >>> tuple_dataset = TupleDataset([0, 1, 2], [0, 1, 4])
       >>> targets = tuple_dataset.features[:, 1]
       >>> print('targets', targets)  # We can extract only target value
       targets [0, 1, 4]

    """
    if self._features_indexer is None:
        self._features_indexer = TupleDatasetFeatureIndexer(self)
    return self._features_indexer


class DictDatasetFeaturesIndexer(BaseFeatureIndexer):
    """FeatureIndexer for DictDataset"""

    def __init__(self, dataset):
        """

        Args:
            dataset (DictDataset): DictDataset instance
        """
        if not isinstance(dataset, DictDataset):
            raise TypeError('dataset class {} is not expected'
                            .format(type(dataset)))
        super(DictDatasetFeaturesIndexer, self).__init__(dataset)
        self.datasets = dataset._datasets

    # Override method to support key-based feature index accessing
    def create_feature_index_list(self, feature_index):
        if isinstance(feature_index, slice):
            raise TypeError('Accessing feature by slice is not supported')
        elif isinstance(feature_index, (list, numpy.ndarray)):
            feature_index_list = feature_index
        else:
            feature_index_list = [feature_index]
        return feature_index_list

    # Override method to check key-based feature index accessing
    def check_type_feature_index(self, j):
        if j not in self.datasets.keys():
            raise IndexError('index {} is not found in feature_keys '
                             .format(j))

    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]


@property
def dd_features(self):
    """Extract features according to the specified index.

    - axis 0 is used to specify dataset id (`i`-th dataset)
    - axis 1 is used to specify feature label

    .. admonition:: Example

       >>> from chainer.datasets import DictDataset
       >>> dd = DictDataset(x=[0, 1, 2], t=[0, 1, 4])
       >>> targets = dd.features[:, 't']
       >>> print('targets', targets)  # We can extract only target value
       targets [0, 1, 4]

    """
    if self._features_indexer is None:
        self._features_indexer = DictDatasetFeaturesIndexer(self)
    return self._features_indexer


class DatasetMixinFeatureIndexer(BaseFeatureIndexer):
    """FeaturesIndexer for DatasetMixin"""

    def __init__(self, dataset):
        """

        Args:
            dataset (DatasetMixin): DatasetMixin instance
        """
        if not isinstance(dataset, DatasetMixin):
            raise TypeError('dataset class {} is not expected'
                            .format(type(dataset)))
        super(DatasetMixinFeatureIndexer, self).__init__(dataset)
        self.feature_cache = None

    def features_length(self):
        return self.dataset.features_length()

    def extract_feature_by_slice(self, slice_index, j):
        return self.dataset.extract_feature_by_slice(slice_index, j)

    def extract_feature(self, i, j):
        return self.dataset.extract_feature(i, j)

    def preprocess(self, item):
        self.dataset.preprocess_extract_feature(item)

    def postprocess(self, item):
        self.dataset.postprocess_extract_feature(item)


@property
def dm_features_length(self):
    """Feature size

    It should return the number of variables returned by `get_example`.

    """
    raise NotImplementedError


def dm_extract_feature_by_slice(self, slice_index, j):
    """This method may be override to support efficient feature extraction.

    If not override, `ExtractBySliceNotSupportedError` is raised by default, 
    and in this case `extract_feature` is used instead.

    Args:
        slice_index (slice): slice of data index to be extracted
        j (int): `j`-th feature to be extracted

    Returns: feature

    """
    raise ExtractBySliceNotSupportedError


def dm_extract_feature(self, i, j):
    """Extracts `i`-th data's `j`-th feature

    This method may be override to support efficient feature extraction.

    Args:
        i (int): `i`-th data to be extracted
        j (int): `j`-th feature to be extracted

    Returns: feature

    """
    if self._features_indexer._extract_single_feature:
        data = self.get_example(i)
    else:
        if i not in self._cache_features:
            # logger = getLogger(__name__)
            # logger.debug('[DEBUG] caching features...')
            self._cache_features[i] = self.get_example(i)
        data = self._cache_features[i]
    if isinstance(data, tuple):
        return data[j]
    elif j == 0:
        return data
    else:
        raise ValueError('[Error] unexpected behavior')


def dm_preprocess_extract_feature(self, item):
    self._cache_features = {}


def dm_postprocess_extract_feature(self, item):
    del self._cache_features


@property
def dm_features(self):
    """Extract features according to the specified index.

    - axis 0 is used to specify dataset id (`i`-th dataset)
    - axis 1 is used to specify feature id (`j`-th feature)

    .. admonition:: Example
    TODO

    """
    if self._features_indexer is None:
        self._features_indexer = DatasetMixinFeatureIndexer(self)
    return self._features_indexer


def cd_features_length(self):
    """
    Args:
        self (ConcatenatedDataset): 
    """
    return self._datasets[0].features.features_length()


def id_features_length(self):
    """features_length for ImageDataset
    Args:
        self (ImageDataset): 
    """
    return 1


def lid_extract_feature(self, i, j):
    """extract_feature for LabeledImageDataset
    
    Args:
        self (LabeledImageDataset): 
        i (int): 
        j (int): 
    """
    if j == 1:
        # Extract label feature
        int_label = self._pairs[i]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return label
    else:
        return self.get_example(i)[j]


def lid_features_length(self):
    """features_length for LabeledImageDataset
    Args:
        self (LabeledImageDataset): 
    """
    return 2


def sd_features_length(self):
    """features_length for SubDataset
    Args:
        self (SubDataset): 
    """
    return self._dataset.features.features_length()


def trd_features_length(self):
    """features_length for TransformDataset
    Args:
        self (TransformDataset): 
    """
    return self._dataset.features.features_length()


def install_features_indexer():
    logger = getLogger(__name__)
    logger.info('installing features indexer...')

    # --- TupleDataset ---
    TupleDataset._features_indexer = None
    TupleDataset.features = td_features
    # --- DictDataset ---
    DictDataset._features_indexer = None
    DictDataset.features = dd_features

    # --- DatasetMixin ---
    DatasetMixin._features_indexer = None
    DatasetMixin.features = dm_features

    DatasetMixin.features_length = dm_features_length
    DatasetMixin.extract_feature_by_slice = dm_extract_feature_by_slice
    DatasetMixin.extract_feature = dm_extract_feature
    DatasetMixin.preprocess_extract_feature = dm_preprocess_extract_feature
    DatasetMixin.postprocess_extract_feature = dm_postprocess_extract_feature

    ConcatenatedDataset.features_length = cd_features_length
    ImageDataset.features_length = id_features_length
    LabeledImageDataset.extract_feature = lid_extract_feature
    LabeledImageDataset.features_length = lid_features_length
    SubDataset.features_length = sd_features_length
    TransformDataset.features_length = trd_features_length


if __name__ == '__main__':
    # install_features_indexer()

    td = TupleDataset([0, 1, 2], [0, 1, 4])
    targets = td.features[:, 1]
    print('td targets', targets)  # We can extract only target value

    dd = DictDataset(x=[0, 1, 2], t=[0, 8, 5])
    targets = dd.features[:, 't']
    print('dd targets', targets)  # We can extract only target value

    # --- DatasetMixin ---
    class SimpleDataset(DatasetMixin):
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def get_example(self, i):
            return self.values[i]

        def features_length(self):
            return 1

    ds1 = SimpleDataset([0, 1, 2, 3, 4, 5])
    ds2 = SimpleDataset([10, 11, 12])
    x = ds1.features[1:4, 0]
    print('ds x', x)

    # --- ConcatenatedDataset ---
    cd = ConcatenatedDataset(ds1, ds2)
    x = cd.features[4:8, 0]
    print('cd x', x)

    # --- SubDataset ---
    from chainer.datasets.sub_dataset import split_dataset_n
    sd1, sd2 = split_dataset_n(ds1, 2)
    x = sd1.features[1:3]
    print('sd x', x)

    # --- TransformDataset ---
    def transform(in_data):
        return in_data * 100

    trd = TransformDataset(cd, transform=transform)
    x = trd.features[4:8]
    print('trd x', x)
