from setuptools import setup

setup(name='essHIC',
      version='1.4',
      description='essHIC is a python module to compare HiC matrices by computing a metric distance between them',
      url='https://github.com/stefanofranzini/essHIC',
      author='Stefano Franzini',
      author_email='sfranzin@sissa.it',
      license='MIT',
      packages=['essHIC'],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'sklearn'
      ],
      zip_safe=True,
      keywords=['HiC','distance','spectral'])
