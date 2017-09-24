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

    def __init__(self, tag=None):
        #print("init")
        self.tag = self.DEFAULT_TAG if tag is None else tag
        self.t = time()

    def __enter__(self):
        #print("enter")
        self.t = time()
        return self

    def __exit__(self, type, value, traceback):
        t_end = time()
        print('[TimeMeasure] {}: {} sec'.format(self.tag, t_end - self.t))
        #print("exit")

    def start(self):
        self.t = time()

    def update(self, tag=None):
        self.tag = self.DEFAULT_TAG if tag is None else tag
        t_end = time()
        print('[TimeMeasure] {}: {:.6f} sec'.format(self.tag, t_end - self.t))
        self.t = t_end

    # def end(self, tag=None):
    #     self.tag = tag
    #     t_end = time()
    #     print('{}: {} sec'.format(self.tag, t_end - self.time_dict[self.tag]))

if __name__ == '__main__':
    # Demo
    tm = TimeMeasure()
    num_repeat = 10000
    for _ in range(num_repeat):
        a = 5 ** 5
    tm.update('{} calculation'.format(num_repeat))
