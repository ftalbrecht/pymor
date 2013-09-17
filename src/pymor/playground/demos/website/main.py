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


from pymor.analyticalproblems import ThermalBlockProblem
from pymor.demos import thermalblock
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.algorithms import greedy, gram_schmidt_basis_extension
from pymor.tools.vtkio import write_vtk

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


def thermalblock_demo(args):
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']),
                                           parameter_range=(PARAM_MIN, PARAM_MAX))
    discretization, pack = discretize_elliptic_cg(problem, diameter=math.sqrt(2) / args['--grid'])
    shape = (args['YBLOCKS'], args['XBLOCKS'])
#     mu = {'diffusion': np.random.random_sample(shape) + 0.01}
    mu = {'diffusion': np.ones(shape) * 0.1}
    U = discretization.solve(mu)
    write_vtk(pack['grid'], U, 'thermal_block', last_step=0)
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
