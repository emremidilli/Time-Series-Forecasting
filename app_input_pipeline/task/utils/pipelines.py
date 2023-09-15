from io import BytesIO

import numpy as np

from tensorflow.python.lib.io import file_io


def read_npy_file(file_path, dtype):
    '''returns npy file'''
    content = np.load(
        BytesIO(
            file_io.read_file_to_string(
                file_path,
                binary_mode=True)),
        allow_pickle=True)

    content = np.cast[dtype](content)

    return content
