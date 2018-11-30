from setuptools import setup

setup(name='pyflsace',
      version='0.1',
      description='',
      url='',
      author='Ulrich Dobramysl',
      author_email='ulrich.dobramysl@gmail.com',
      packages=['flsace'],
      zip_safe=False,
      test_suite='tests',
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'scikit-image>=0.13.0',
          'scikit-learn',
          'lapjv',
      ])
