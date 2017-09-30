import os


def walk_all_files(directory):
    """return all files inside specified directory recursively, but not directory
    
    Ref: http://qiita.com/suin/items/cdef17e447ceeff6e79d
    
    Args:
        directory (str): directory path

    Returns:

    """
    for root, dirs, files in os.walk(directory):
        # print('root', root, 'dirs', dirs, 'files', files)
        if os.path.isfile(root):
            yield root
        for file in files:
            path = os.path.join(root, file)
            if os.path.isfile(path):
                yield path


def collect_files(directory, file_name):
    """return all files with file name `file_name` inside specified directory.

    Ref: http://qiita.com/suin/items/cdef17e447ceeff6e79d

    Args:
        directory (str): directory path
        file_name (str): file name to search

    Returns:

    """
    for root, dirs, files in os.walk(directory):
        # print('root', root, 'dirs', dirs, 'files', files)
        if file_name in files:
            path = os.path.join(root, file_name)
            if os.path.isfile(path):
                yield path


if __name__ == '__main__':
    import os
    mode = 2
    if mode == 1:
        for path in walk_all_files('.'):
            print(path, os.path.abspath(path))
    elif mode == 2:
        for path in collect_files('.', 'args'):
            print('collect_files args:', path, os.path.abspath(path))
