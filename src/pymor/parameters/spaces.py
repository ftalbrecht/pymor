# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, product
import numpy as np

from pymor.parameters.base import Parameter, ParameterType
from pymor.parameters.interfaces import ParameterSpaceInterface
from pymor.tools import new_random_state


class CubicParameterSpace(ParameterSpaceInterface):
    """Simple |ParameterSpace| where each summand is an n-cube.

    Parameters
    ----------
    parameter_type
        The |ParameterType| of the space.
    minimum
        The minimum for each matrix entry of each |Parameter| component.
        Must be `None` if `ranges` is not `None`.
    maximum
        The maximum for each matrix entry of each |Parameter| component.
        Must be `None` if `ranges` is not `None`.
    ranges
        dict whose keys agree with `parameter_type` and whose values
        are tuples (min, max) specifying the minimum and maximum of each
        matrix entry of corresponding |Parameter| component.
        Must be `None` if `minimum` and `maximum` are specified.
    """

    def __init__(self, parameter_type, minimum=None, maximum=None, ranges=None):
        assert ranges is None or (minimum is None and maximum is None), 'Must specify minimum, maximum or ranges'
        assert ranges is not None or (minimum is not None and maximum is not None),\
            'Must specify minimum, maximum or ranges'
        assert minimum is None or minimum < maximum
        parameter_type = ParameterType(parameter_type)
        self.parameter_type = parameter_type
        self.ranges = {k: (minimum, maximum) for k in parameter_type} if ranges is None else ranges

    def parse_parameter(self, mu):
        return Parameter.from_parameter_type(mu, self.parameter_type)

    def contains(self, mu):
        mu = self.parse_parameter(mu)
        return all(np.all(self.ranges[k][0] <= mu[k]) and np.all(mu[k] <= self.ranges[k][1])
                   for k in self.parameter_type)

    def sample_uniformly(self, counts):
        """Iterator sampling uniformly |Parameters| from the space."""
        if isinstance(counts, dict):
            pass
        elif isinstance(counts, (tuple, list, np.ndarray)):
            counts = {k: c for k, c in izip(self.parameter_type, counts)}
        else:
            counts = {k: counts for k in self.parameter_type}
        linspaces = tuple(np.linspace(self.ranges[k][0], self.ranges[k][1], num=counts[k]) for k in self.parameter_type)
        iters = tuple(product(ls, repeat=max(1, np.zeros(sps).size))
                      for ls, sps in izip(linspaces, self.parameter_type.values()))
        for i in product(*iters):
            yield Parameter(((k, np.array(v).reshape(shp))
                             for k, v, shp in izip(self.parameter_type, i, self.parameter_type.values())))

    def sample_randomly(self, count=None, random_state=None, seed=None):
        """Iterator sampling random |Parameters| from the space."""
        assert not random_state or seed is None
        c = 0
        ranges = self.ranges
        random_state = random_state or new_random_state(seed)
        while count is None or c < count:
            yield Parameter(((k, random_state.uniform(ranges[k][0], ranges[k][1], shp))
                             for k, shp in self.parameter_type.iteritems()))
            c += 1

    def __str__(self):
        rows = [(k, str(v), str(self.ranges[k])) for k, v in self.parameter_type.iteritems()]
        column_widths = [max(map(len, c)) for c in zip(*rows)]
        return ('CubicParameterSpace\n' +
                '\n'.join(('key: {:' + str(column_widths[0] + 2)
                           +'} shape: {:' + str(column_widths[1] + 2)
                           + '} range: {}').format(c1 + ',', c2 + ',', c3) for (c1, c2, c3) in rows))

    def __repr__(self):
        return 'CubicParameterSpace({}, ranges={})'.format(repr(self.parameter_type), repr(self.ranges))
