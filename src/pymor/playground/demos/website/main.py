#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bottle import (route, run, debug, PasteServer, static_file, redirect, abort, request, default_app)
import os
from jinja2 import Environment, FileSystemLoader
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
import pprint

from pymor.demos import thermalblock

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

@route('/index')
def index():
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
    errs, err_mus, ests, est_mus, conds, cond_mus = thermalblock.thermalblock_demo(args)
    template = env.get_template('index.html')
    ret = template.render(errs=pprint.pformat(errs))
    return ret

if __name__ == "__main__":
    port = 6666
    debug(True)
    app = default_app()
    run(app=app, server=PasteServer, host='localhost', port=port , reloader=False)
