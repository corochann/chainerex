import chainerex


# Flag which checks if the install_indexers are already called or not.
CHAINEREX_INSTALL_INDEXERS = False


def install_indexers():
    global CHAINEREX_INSTALL_INDEXERS
    if not CHAINEREX_INSTALL_INDEXERS:
        chainerex.dataset.indexers.feature_indexer.install_features_indexer()
        CHAINEREX_INSTALL_INDEXERS = True
