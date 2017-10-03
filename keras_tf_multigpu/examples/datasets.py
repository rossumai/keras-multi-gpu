import numpy as np
from keras.utils import to_categorical

def create_synth_dataset(image_size, class_count, dataset_size):
    input_shape = (image_size, image_size, 3)
    X = np.random.rand(*((dataset_size,) + input_shape))
    y = np.random.randint(low=0, high=class_count, size=dataset_size)
    y = to_categorical(y, class_count)
    return X, y

def create_synth_imagenet(image_size, dataset_size):
    # image_size: typically 224 or 299
    return create_synth_dataset(image_size=image_size, class_count=1000, dataset_size=dataset_size)

def create_synth_cifar10(dataset_size):
    return create_synth_dataset(image_size=32, class_count=10, dataset_size=dataset_size)
