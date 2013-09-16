#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bottle import (route, run, debug, PasteServer, static_file, redirect, abort, request, default_app)
import os
from jinja2 import Environment, FileSystemLoader
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
import pprint
import numpy as np
import math
from pyvtk import (VtkData, UnstructuredGrid, Vectors, PointData, Scalars)

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.demos import thermalblock
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.algorithms import greedy, gram_schmidt_basis_extension
from pymor.grids.referenceelements import triangle, square

PARAM_STEPS = 10
PARAM_MIN = 0.1
PARAM_MAX = 1

cache_opts = {
    'cache.type': 'memory',
    'cache.data_dir': 'tmp/cache/data',
    'cache.lock_dir': 'tmp/cache/lock'
}
env = Environment(loader=FileSystemLoader('.'))
cache = CacheManager(**parse_cache_config_options(cache_opts))

@route('/static/:filename')
def statics(filename):
    return static_file(filename, root=os.getcwd() + '/static/')

def triangle_data_to_vtk(subentity_ordering, coords, data, filename):
    subs = subentity_ordering
    num_points = len(coords[0])
    points = [[coords[0][i], coords[1][i], coords[2][i]] for i in xrange(num_points)]
    padded_data = [data[i] for i in xrange(num_points) ]
    dummy = Vectors([[1, 1, 1] for _ in xrange(num_points)])
    us_grid = UnstructuredGrid(points, triangle=subs)
    pd = PointData(dummy, Scalars(padded_data))

    vtk = VtkData(us_grid, pd, 'Unstructured Grid Example')
    vtk.tofile(filename)
    vtk.tofile(filename + '_bin', 'binary')


def write_vtk(grid, data, bb=[[0, 0], [1, 1]]):
    size = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])
    scale = 1 / size
    shift = -np.array(bb[0]) - size / 2
    if grid.reference_element == triangle:
        x, y = (grid.centers(2)[:, 0] + shift[0]) * scale[0], (grid.centers(2)[:, 1] + shift[1]) * scale[1]
        z = np.zeros(len(x))
        triangle_data_to_vtk(grid.subentities(0, 2).tolist(), (x, y, z), data._array[0, :], 'thermal_block')
    else:
        raise Exception()

def thermalblock_demo(args):
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']),
                                           parameter_range=(PARAM_MIN, PARAM_MAX))
    discretization, pack = discretize_elliptic_cg(problem, diameter=math.sqrt(2) / args['--grid'])
    shape = (args['YBLOCKS'], args['XBLOCKS'])
    mu = {'diffusion': np.random.random_sample(shape) + 0.01}
    mu = {'diffusion': np.ones(shape) * 0.1}
    U = discretization.solve(mu)
    write_vtk(pack['grid'], U)
    return None

def default_args():
    args = {}
    args['XBLOCKS'] = 2
    args['YBLOCKS'] = 2
    args['--grid'] = 20
    args['SNAPSHOTS'] = 5
    args['RBSIZE'] = 5
    args['--test'] = 5
    args['--estimator-norm'] = 'h1'
    args['--extension-alg'] = 'numpy_trivial'
    args['--reductor'] = 'default'
    args['--with-estimator'] = True
    args['--plot-solutions'] = args['--plot-err'] = args['--plot-error-sequence'] = False
    return args

@route('/index')
def index():
    errs = thermalblock_demo(default_args())
    template = env.get_template('index.html')
    ret = template.render(errs=pprint.pformat(errs))
    return ret

if __name__ == "__main__":
    port = 6666
#     debug(True)
#     app = default_app()
#     run(app=app, server=PasteServer, host='localhost', port=port , reloader=False)
    thermalblock_demo(default_args())
