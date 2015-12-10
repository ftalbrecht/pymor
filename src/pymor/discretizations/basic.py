# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import chain

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.la.basic import induced_norm
from pymor.la.interfaces import VectorArrayInterface
from pymor.operators.constructions import VectorOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.base import Parameter
from pymor.tools.arguments import method_arguments
from pymor.tools.frozendict import FrozenDict


class DiscretizationBase(DiscretizationInterface):
    """Base class for |Discretizations| providing some common functionality."""

    def __init__(self, operators, functionals, vector_operators, products=None, estimator=None, visualizer=None,
                 cache_region='disk', name=None):
        self.operators = FrozenDict(operators)
        self.functionals = FrozenDict(functionals)
        self.vector_operators = FrozenDict(vector_operators)
        self.linear = all(op is None or op.linear for op in chain(operators.itervalues(), functionals.itervalues()))
        self.products = products
        self.estimator = estimator
        self.visualizer = visualizer
        self.cache_region = cache_region
        self.name = name

        if products:
            for k, v in products.iteritems():
                setattr(self, '{}_product'.format(k), v)
                setattr(self, '{}_norm'.format(k), induced_norm(v))
        if estimator is not None:
            self.estimate = self.__estimate
        if visualizer is not None:
            self.visualize = self.__visualize

    def __visualize(self, U, *args, **kwargs):
        self.visualizer.visualize(U, self, *args, **kwargs)

    def __estimate(self, U, mu=None):
        return self.estimator.estimate(U, mu=mu, discretization=self)

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'estimate' in d:
            del d['estimate']
        if 'visualize' in d:
            del d['visualize']
        return d

    def __setstate__(self, s):
        if s['estimator']:
            s['estimate'] = self.__estimate
        if s['visualizer']:
            s['visualize'] = self.__visualize
        self.__dict__.update(s)


class StationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a linear functional F and a (possibly) non-linear operator L.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The |Functional| F.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the discretization.
    functionals
        A dict of (output) |Functionals| which can be evaluated on solution
        vectors.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None` an `estimate(U, mu)` method is added to the
        discretization.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None` a `visualize(U, *args, **kwargs)` method is added
        to the discretization, which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the discretization.

    Attributes
    ----------
    operator
        The |Operator| L. Synonymous for `operators['operator']`.
    rhs
        The |Functional| F. Synonymous for `functionals['rhs']`.
    """

    sid_ignore = ('visualizer', 'cache_region', 'name')

    def __init__(self, operator, rhs, products=None, functionals=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region='disk', name=None):
        functionals = functionals or {}
        assert isinstance(operator, OperatorInterface)
        assert isinstance(rhs, OperatorInterface) and rhs.linear
        assert operator.source == operator.range == rhs.source
        assert rhs.range.dim == 1
        assert all(f.source == operator.source for f in functionals.values())
        assert 'rhs' not in functionals or rhs == functionals['rhs']

        operators = {'operator': operator}
        functionals_with_rhs = {'rhs': rhs}
        functionals_with_rhs.update(functionals)
        super(StationaryDiscretization, self).__init__(operators=operators,
                                                       functionals=functionals_with_rhs,
                                                       vector_operators={}, products=products, estimator=estimator,
                                                       visualizer=visualizer, cache_region=cache_region, name=name)
        self.solution_space = operator.source
        self.operator = operator
        self.rhs = rhs
        self.operators = operators
        self.build_parameter_type(inherits=(operator, rhs))
        self.parameter_space = parameter_space

    with_arguments = frozenset(method_arguments(__init__)).union({'operators', 'vector_operators'})

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs \
            or 'operator' not in kwargs and kwargs['operators'].keys() == ['operator']

        assert 'vector_operators' not in kwargs or not kwargs['vector_operators'].keys()

        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))
        if 'rhs' not in kwargs and 'functionals' in kwargs and 'rhs' in kwargs['functionals']:
            kwargs['rhs'] = kwargs['functionals']['rhs']
        elif 'rhs' in kwargs and 'functionals' not in kwargs:
            functionals = dict(self.functionals)
            functionals['rhs'] = kwargs['rhs']
            kwargs['functionals'] = functionals
        if 'vector_operators' in kwargs:
            del kwargs['vector_operators']

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)

        # explicitly checking if logging is disabled saves the str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.operator.apply_inverse(self.rhs.as_vector(mu), mu=mu)


class InstationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of stationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly) non-linear time-dependent
    |Operator|, F is a time-dependent linear |Functional|, and u_0 the
    initial data. The mass operator is assumed to be linear,
    time-independent and |Parameter|-independent.

    Parameters
    ----------
    T
        The end-time T.
    initial_data
        The initial data u_0. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`).
    operator
        The |Operator| L.
    rhs
        The |Functional| F.
    mass
        The mass |Operator| `M`. If `None` the identity is assumed.
    time_stepper
        T time-stepper to be used by :meth:`solve`. Has to satisfy
        the :class:`~pymor.algorithms.timestepping.TimeStepperInterface`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the discretization.
    functionals
        A dict of (output) |Functionals| which can be evaluated on solution
        vectors.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None` an `estimate(U, mu)` method is added to the
        discretization.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None` a `visualize(U, *args, **kwargs)` method is added
        to the discretization, which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the discretization.

    Attributes
    ----------
    T
        The end-time.
    initial_data
        The intial data u_0 given by a vector-like |Operator|. Synonymous
        for `vector_operators['initial_data']`.
    operator
        The |Operator| L. Synonymous for `operators['operator']`.
    rhs
        The |Functional| F. Synonymous for `functionals['rhs']`.
    mass
        The mass operator M. Synonymous for `operators['mass']`.
    time_stepper
        The provided time-stepper.
    """

    sid_ignore = ('visualizer', 'cache_region', 'name')

    def __init__(self, T, initial_data, operator, rhs=None, mass=None, time_stepper=None, num_values=None,
                 products=None, functionals=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region='disk', name=None):
        functionals = functionals or {}
        if rhs is None and 'rhs' in functionals:
            rhs = functionals['rhs']
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = VectorOperator(initial_data, name='initial_data')
        assert isinstance(initial_data, OperatorInterface)
        assert not isinstance(initial_data, OperatorInterface) or initial_data.source.dim == 1
        assert isinstance(operator, OperatorInterface)
        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert mass is None or isinstance(mass, OperatorInterface) and mass.linear
        assert isinstance(time_stepper, TimeStepperInterface)
        # assert operator.source == operator.range == initial_data.range
        assert rhs is None or rhs.source == operator.source and rhs.range.dim == 1
        assert mass is None or mass.source == mass.range == operator.source
        assert all(f.source == operator.source for f in functionals.values())
        assert 'rhs' not in functionals or rhs == functionals['rhs']

        operators = {'operator': operator, 'mass': mass}
        functionals_with_rhs = {'rhs': rhs} if rhs else {}
        functionals_with_rhs.update(functionals)
        vector_operators = {'initial_data': initial_data}
        super(InstationaryDiscretization, self).__init__(operators=operators, functionals=functionals_with_rhs,
                                                         vector_operators=vector_operators,
                                                         products=products, estimator=estimator,
                                                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.solution_space = operator.source
        self.initial_data = initial_data
        self.operator = operator
        self.rhs = rhs
        self.mass = mass
        self.time_stepper = time_stepper
        self.num_values = num_values
        self.build_parameter_type(inherits=(initial_data, operator, rhs, mass), provides={'_t': 0})
        self.parameter_space = parameter_space

        if hasattr(time_stepper, 'nt'):
            self.with_arguments = self.with_arguments.union({'time_stepper_nt'})

    with_arguments = frozenset(method_arguments(__init__)).union({'operators', 'functionals', 'vector_operators'})

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs or kwargs['operators'].viewkeys() <= {'operator', 'mass'}
        assert 'vector_operators' not in kwargs or kwargs['vector_operators'].viewkeys() <= {'initial_data'}
        assert 'operators' not in kwargs or not set(kwargs['operators']).intersection(kwargs.viewkeys())
        assert 'vector_operators' not in kwargs or not set(kwargs['vector_operators']).intersection(kwargs.viewkeys())
        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs
        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))
        if 'rhs' not in kwargs and 'functionals' in kwargs and 'rhs' in kwargs['functionals']:
            kwargs['rhs'] = kwargs['functionals']['rhs']
        elif 'rhs' in kwargs and 'functionals' not in kwargs:
            functionals = dict(self.functionals)
            functionals['rhs'] = kwargs['rhs']
            kwargs['functionals'] = functionals
        if 'vector_operators' in kwargs:
            kwargs.update(kwargs.pop('vector_operators'))
        if 'time_stepper_nt' in kwargs:
            kwargs['time_stepper'] = self.time_stepper.with_(nt=kwargs.pop('time_stepper_nt'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu).copy() if self.parametric else Parameter({})

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_vector(mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                       initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
