import numpy as np
import torch
import torch.nn as nn

tanh = nn.Tanh()
selu = nn.SELU()
sigmoid = nn.Sigmoid()

def non_linear_fn(t, x, y, z):
  # returns a function from variables
  return tanh(torch.tensor(20*(t - 2*(x-.5)))) + selu(torch.tensor((t-2*(y-0.5)))) + sigmoid(torch.tensor(-20*(t-(z-0.5))))

def generate_data(values, function=non_linear_fn, length=25, range_=[-1, 1]):
    """
    Function to generate data from values

    :param values: values to function for generating spectra
    :type values: float
    :param function:  mathematical expression used to generate spectra
    :type function: function, optional
    :param length: spectral length
    :type length: int (optional)
    :param range_: x range for function
    :type range_:  list of float
    :return: generatered spectra
    :rtype: array of float
    """

    # build x vector
    x = np.linspace(range_[0], range_[1], length)

    data = np.zeros((values.shape[0], length))

    for i in range(values.shape[0]):
        data[i, :] = function(x, values[i, 0], values[i, 1], values[i, 2])

    return data