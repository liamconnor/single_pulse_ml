from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='single_pulse_ml',
      version='0.1',
      description='Machine learning implementation of single-pulse search',
      url='http://github.com/liamconnor/single_pulse_ml',
      author='Flying Circus',
      author_email='liam.dean.connor@gmail.com',
      license='MIT',
      packages=['single_pulse_ml'],
      install_requires=[
          'sklearn',
          'tensorflow'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
