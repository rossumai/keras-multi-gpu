import keras.backend as K

if K.backend() != 'tensorflow':
    raise RuntimeError('This package only supports TensorFlow backend.')
