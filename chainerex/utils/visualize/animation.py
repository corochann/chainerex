"""
Ref: http://own-search-and-study.xyz/2017/05/18/python%E3%81%AEmatplotlib%E3%81%A7gif%E3%82%A2%E3%83%8B%E3%83%A1%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B/
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from tqdm import tqdm


def png_to_gif(png_filepath_list, save_gif_filepath='animation.gif',
               fps=None, writer=None):
    fig = plt.figure()

    # delete axis
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x', which='both', top='off', bottom='off',
                   labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off',
                   labelleft='off')

    # create empty list
    ims = []

    # read image png files
    for i in tqdm(range(len(png_filepath_list))):
        tmp = Image.open(png_filepath_list[i])

        # Linear interpolation to prevent aliasing
        ims.append([plt.imshow(tmp, interpolation="spline36")])

    # animation
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
    ani.save(save_gif_filepath, fps=fps, writer=writer)


if __name__ == '__main__':
    # This example shows to create gif file animation from png file under the
    # `dir_path` directory
    from glob import glob
    dir_path = 'hoge'
    save_path = os.path.join(dir_path, 'anim.gif')
    png_filepath_list = glob('{}/*.png'.format(dir_path))
    png_filepath_list.sort()
    print(png_filepath_list)
    png_to_gif(png_filepath_list,
               save_gif_filepath=save_path,
               fps=1,
               writer='imagemagick')
