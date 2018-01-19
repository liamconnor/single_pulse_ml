from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='single_pulse_ml',
      version='0.1',
      description='Machine learning implementation of single-pulse search',
      url='http://github.com/liamconnor/single_pulse_ml',
      author='Liam Connor',
      author_email='liam.dean.connor@gmail.com',
      license='GPL v2.0',
      packages=['single_pulse_ml'],
      install_requires=[
          'numpy',
          'scipy',
          'h5py',
          'matplotlib',
          'sklearn',
          'tensorflow',
          'keras',
          ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
