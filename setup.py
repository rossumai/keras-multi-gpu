from setuptools import setup

setup(name='keras_tf_multigpu',
      version='0.1',
      description='Multi-GPU data-parallel training in Keras/TensorFlow',
      url='https://github.com/rossumai/keras-multi-gpu',
      author='Bohumir Zamecnik',
      author_email='bohumir.zamecnik@gmail.com',
      license='MIT',
      packages=['keras_tf_multigpu'],
      zip_safe=False,
      install_requires=[
         'Keras>=2.0.8',
         'numpy',
         'tensorflow-gpu>=1.3',
      ],
      setup_requires=['setuptools-markdown'],
      long_description_markdown_filename='README.md',
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',

          'Operating System :: POSIX :: Linux',
      ])
