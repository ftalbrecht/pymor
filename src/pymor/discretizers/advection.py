# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np

from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.core import inject_sid
from pymor.discretizations import InstationaryDiscretization
from pymor.domaindiscretizers import discretize_domain_default
from pymor.grids import RectGrid
from pymor.gui.qt import GlumpyPatchVisualizer, Matplotlib1DVisualizer
from pymor.la import NumpyVectorArray
from pymor.la import induced_norm
from pymor.operators import NumpyMatrixOperator
from pymor.operators.fv import (nonlinear_advection_lax_friedrichs_operator,
                                nonlinear_advection_engquist_osher_operator,
                                nonlinear_advection_simplified_engquist_osher_operator,
                                L2Product)


def discretize_nonlinear_instationary_advection_fv(analytical_problem, diameter=None, nt=100, num_flux='lax_friedrichs',
                                                   lxf_lambda=1., eo_gausspoints=5, eo_intervals=1, num_values=None,
                                                   domain_discretizer=None, grid=None, boundary_info=None):

    assert isinstance(analytical_problem, InstationaryAdvectionProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert num_flux in ('lax_friedrichs', 'engquist_osher', 'simplified_engquist_osher')

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if num_flux == 'lax_friedrichs':
        L = nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, p.flux_function,
                                                        dirichlet_data=p.dirichlet_data, lxf_lambda=lxf_lambda)
    elif num_flux == 'engquist_osher':
        L = nonlinear_advection_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                        p.flux_function_derivative,
                                                        gausspoints=eo_gausspoints, intervals=eo_intervals,
                                                        dirichlet_data=p.dirichlet_data)
    else:
        L = nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                                   p.flux_function_derivative,
                                                                   dirichlet_data=p.dirichlet_data)
    F = None if p.rhs is None else L2ProductFunctional(grid, p.rhs)
    I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
    I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
    I = NumpyVectorArray(I)
    inject_sid(I, __name__ + '.discretize_nonlinear_instationary_advection_fv.initial_data', p.initial_data, grid)

    products = {'l2': L2Product(grid, boundary_info)}
    if grid.dim == 2:
        visualizer = GlumpyPatchVisualizer(grid=grid, bounding_box=grid.domain, codim=0)
    elif grid.dim == 1:
        visualizer = Matplotlib1DVisualizer(grid, codim=0)
    else:
        visualizer = None
    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None
    time_stepper = ExplicitEulerTimeStepper(nt=nt)

    discretization = InstationaryDiscretization(operator=L, rhs=F, initial_data=I, T=p.T, products=products,
                                                time_stepper=time_stepper,
                                                parameter_space=parameter_space, visualizer=visualizer,
                                                num_values=num_values, name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
