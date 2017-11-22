"""
Ref: http://shin.hateblo.jp/entry/2013/03/23/211750

Usage 1. tm.start() is ued to start, and tm.update('tag') is to show time with
         'tag' name.

    tm = TimeMeasure()

    tm.start()
    # Run method1
    tm.update('method1')
    # Run method2
    tm.update('method2')
    ...
    tm.start()
    # Run method3
    tm.update('method3')


Usage 2. with ... statement

    with TimeMeasure('tag'):
        # Do something here with time measured


"""
from time import time


class TimeMeasure:

    DEFAULT_TAG = 'time'

    def __init__(self, tag=None, loglevel=5):
        """
        
        Args:
            tag (str or None): 
            loglevel (int): 
               - 0 -> not print
               - 1 -> error
               - 2 -> warning
               - 3 -> info
               - 4 -> debug
               - 5 -> verbose
        """
        self.tag = self.DEFAULT_TAG if tag is None else tag
        self.loglevel = loglevel

        # -- initalize --
        self.tag_count_dict = {}
        self.tag_time_dict = {}
        self.t = time()

    def _update_tag_dict(self, tag, t):
        if self.tag in self.tag_time_dict.keys():
            self.tag_count_dict[tag] += 1
            self.tag_time_dict[tag] += t
        else:
            self.tag_count_dict[tag] = 1
            self.tag_time_dict[tag] = t

    def __enter__(self):
        self.t = time()
        return self

    def __exit__(self, type, value, traceback):
        t_end = time()
        if self.loglevel >= 3:
            print('[TimeMeasure] {}: {} sec'.format(self.tag, t_end - self.t))
        self._update_tag_dict(self.tag, t_end - self.t)

    def start(self):
        self.t = time()

    def update(self, tag=None):
        self.tag = self.DEFAULT_TAG if tag is None else tag
        t_end = time()
        if self.loglevel >= 5:
            print('[TimeMeasure] {}: {:.6f} sec'.format(self.tag, t_end - self.t))
        self._update_tag_dict(self.tag, t_end - self.t)
        self.t = t_end

    def show_stats(self):
        if self.loglevel >= 4:
            for k in self.tag_time_dict.keys():
                print('[TimeMeasure.show_stats] {}: {:.6f} sec / {:6} count'
                      .format(k, self.tag_time_dict[k], self.tag_count_dict[k]))

    def show_average(self):
        if self.loglevel >= 4:
            for k in self.tag_time_dict.keys():
                t = self.tag_time_dict[k] / self.tag_count_dict[k]
                print('[TimeMeasure.show_average] {}: {:.6f} sec'.format(k, t))


    # def end(self, tag=None):
    #     self.tag = tag
    #     t_end = time()
    #     print('{}: {} sec'.format(self.tag, t_end - self.time_dict[self.tag]))

if __name__ == '__main__':
    # Demo
    tm = TimeMeasure(loglevel=4)
    num_repeat = 10000
    for _ in range(num_repeat):
        a = 5 ** 5
    tm.update('{} calculation'.format(num_repeat))

    tm.update('hoge')
    tm.update('hoge')
    tm.update('hoge')
    tm.show_stats()
    tm.show_average()
