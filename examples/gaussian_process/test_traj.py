#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import SquareExpWithBool
from sklearn.datasets import fetch_mldata
import sys

# read training data from the data source
X
