class ExtractBySliceNotSupportedError(Exception):
    pass


class BaseIndexer(object):
    """Base class for Indexer"""

    def __getitem__(self, item):
        raise NotImplementedError
