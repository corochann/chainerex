from distutils.core import setup

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer>=2.0',
]

setup(name='chainerex',
      version='0.0.1',
      description='Unofficial experimental, extra chainer package.',
      packages=find_packages(),
      author='corochann',
      author_email='corochannz@gmail.com',
      url='https://github.com/corochann/chainerex',
      license='MIT',
      setup_requires=setup_requires,
      install_requires=install_requires
)
