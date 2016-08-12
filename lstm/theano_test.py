#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 8/12/16.
#  Copyright (c) 2016 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


import numpy as np
import theano
from theano import config
import theano.tensor as tensor

if __name__ == "__main__":
    x = np.array([1., 2., 3.] * 3, config.floatX)
    y = np.array([3., 4., 5.], np.int64)
    z = np.vstack((x, x.copy()))
    x = theano.shared(x)
    y = theano.shared(y)
    z_in = tensor.tensor3('z', dtype=config.floatX)
    x_in = tensor.matrix('x', dtype=config.floatX)
    y_in = tensor.vector('y', dtype=np.int64.__name__)
    x_dot_y = tensor.dot(x_in, y_in)
    lr = tensor.scalar(name='lr')
    pass
