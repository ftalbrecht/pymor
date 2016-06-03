# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
from itertools import izip

import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.operators.block import BlockOperator
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.reductors.basic import GenericRBReconstructor


class GenericBlockRBReconstructor(BasicInterface):
    """Block variant of GenericRBReconstructor"""

    def __init__(self, RB):
        assert isinstance(RB, (tuple, list))
        self.RB = tuple(RB)

    def reconstruct(self, U):
        if U.dim == 0:
            return BlockVectorArray([GenericRBReconstructor(self.RB[ii].empty()).reconstruct(U)
                                    for ii in np.arange(len(self.RB))])
        else:
            assert isinstance(U, BlockVectorArray)
            assert all(subspace.type == NumpyVectorArray for subspace in U.space.subtype)
            return BlockVectorArray([GenericRBReconstructor(rb).reconstruct(block)
                                     for rb, block in izip(self.RB, U._blocks)])

    def restricted_to_subbasis(self, dim):
        assert dim <= np.max(len(rb) for rb in self.RB)
        if not isinstance(dim, tuple):
            dim = len(self.RB)*[dim]
        assert all([dd <= len(rb) for dd, rb in izip(dim, self.RB)]) # NOTE: problematic for varying local sizes
        return GenericBlockRBReconstructor([rb.copy(ind=range(dd)) for rb, dd in izip(self.RB, dim)])


def reduce_generic_block_rb(discretization, RB, vector_product=None, disable_caching=True, extends=None, sparse=True):

    assert all(isinstance(op, BlockOperator) if op else True for op in discretization.operators)
    assert all(isinstance(op, BlockOperator) if op else True for op in discretization.functionals)
    assert all(isinstance(op, BlockOperator) if op else True for op in discretization.vector_operators)
    assert extends is None or len(extends) == 3

    if RB is None:
        RB = discretization.solution_space.empty()

    projected_operators = {k: unblock(op.projected(range_basis=RB, source_basis=RB, product=None), sparse) if op else None
                           for k, op in discretization.operators.iteritems()}
    projected_functionals = {k: unblock(f.projected(range_basis=None, source_basis=RB, product=None), sparse) if f else None
                             for k, f in discretization.functionals.iteritems()}
    projected_vector_operators = {k: (unblock(op.projected(range_basis=RB, source_basis=None, product=vector_product),
                                              sparse) if op else None)
                                  for k, op in discretization.vector_operators.iteritems()}

    if discretization.products is not None:
        assert all(isinstance(op, BlockOperator) if op else True for op in discretization.products)
        projected_products = {k: unblock(p.projected(range_basis=RB, source_basis=RB), sparse)
                              for k, p in discretization.products.iteritems()}
    else:
        projected_products = None

    cache_region = None if disable_caching else discretization.caching

    rd = discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators=projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=None,
                              cache_region=cache_region, name=discretization.name + '_reduced')
    rd.disable_logging()
    rc = GenericBlockRBReconstructor(RB)

    def unblock(op, sparse=False):
        assert op._blocks[0][0] is not None
        if isinstance(op._blocks[0][0], LincombOperator):
            coefficients = op._blocks[0][0].coefficients
            operators = [None for kk in np.arange(len(op._blocks[0][0].operators))]
            for kk in np.arange(len(op._blocks[0][0].operators)):
                ops = [[op._blocks[ii][jj].operators[kk]
                        if op._blocks[ii][jj] is not None else None
                        for jj in np.arange(op.num_source_blocks)]
                       for ii in np.arange(op.num_range_blocks)]
                operators[kk] = unblock(BlockOperator(ops))
            return LincombOperator(operators=operators, coefficients=coefficients)
        else:
            assert all(all([isinstance(block, NumpyMatrixOperator) if block is not None else True
                           for block in row])
                       for row in op._blocks)
            if op.source.dim == 0 and op.range.dim == 0:
                return NumpyMatrixOperator(np.zeros((0, 0)))
            elif op.source.dim == 1:
                mat = np.concatenate([op._blocks[ii][0]._matrix
                                      for ii in np.arange(op.num_range_blocks)],
                                     axis=1)
            elif op.range.dim == 1:
                mat = np.concatenate([op._blocks[0][jj]._matrix
                                      for jj in np.arange(op.num_source_blocks)],
                                     axis=1)
            else:
                mat = bmat([[coo_matrix(op._blocks[ii][jj]._matrix)
                             if op._blocks[ii][jj] is not None else coo_matrix((op._range_dims[ii], op._source_dims[jj]))
                             for jj in np.arange(op.num_source_blocks)]
                            for ii in np.arange(op.num_range_blocks)])
                mat = mat.toarray()
            return NumpyMatrixOperator(mat)

    return rd, rc, {}

