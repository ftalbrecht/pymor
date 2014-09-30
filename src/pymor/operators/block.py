# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, bmat

from pymor.operators.interfaces import OperatorInterface
from pymor.la.interfaces import VectorArrayInterface, VectorSpace
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.la.blockvectorarray import BlockVectorArray
from pymor.operators.basic import NumpyMatrixOperator, NumpyMatrixBasedOperator, OperatorBase


def enumerated_matrix_iterator(matrix):
    for i, row in enumerate(matrix):
        for j, entry in enumerate(row):
            yield i, j, entry


class BlockOperator(OperatorBase):

    def __init__(self, blocks, sources=None, ranges=None):
        assert isinstance(blocks, list)
        if isinstance(blocks[0], OperatorInterface):
            assert sources is None and ranges is None
            assert all([isinstance(block, OperatorInterface) for block in blocks])
            assert all([block.range == blocks[0].range for block in blocks])
            if blocks[0].range.dim == 1:
                assert all([block.range.dim == 1 for block in blocks])
                self.source = VectorSpace(BlockVectorArray, [block.source for block in blocks])
                self.range = blocks[0].range
                self._blocks = blocks
                self.num_source_blocks = len(self._blocks)
                self.num_range_blocks = 1
                self.linear = all([block.linear for block in self._blocks])
                # self.invert_options = None
            else:
                raise Exception('Not implemented yet!')
            self.build_parameter_type(inherits=self._blocks)
        else:
            assert sources is not None and ranges is not None
            assert isinstance(sources, list) and isinstance(ranges, list)
            self.source = VectorSpace(BlockVectorArray, sources)
            self.range = VectorSpace(BlockVectorArray, ranges)
            self._range_dims = ranges #[r.dim for r in ranges]
            self._source_dims = sources #[s.dim for s in sources]
            self.num_source_blocks = len(self._source_dims)
            self.num_range_blocks = len(self._range_dims)
            assert len(blocks) == len(self._range_dims)
            assert all([len(block_row) == len(self._source_dims) for block_row in blocks])
            assert all(isinstance(block, OperatorInterface) if block is not None else True for _, _, block in enumerated_matrix_iterator(blocks))
            assert all([all([blocks[ii][jj].source.dim == sources[jj] if blocks[ii][jj] is not None else True for jj in np.arange(self.num_source_blocks)]) for ii in np.arange(self.num_range_blocks)])
            assert all([all([blocks[ii][jj].range.dim == ranges[ii] if blocks[ii][jj] is not None else True for jj in np.arange(self.num_source_blocks)]) for ii in np.arange(self.num_range_blocks)])
            self._blocks = blocks
            self.build_parameter_type(inherits=[block for block in block_row for block_row in blocks])
            self.linear = all([all([block.linear if block is not None else True for block in blocks_row]) for blocks_row in self._blocks])
            # self.invert_options = None

    def apply(self, U, ind=None, mu=None):
        raise Exception('Not implemented yet!')

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        assert V in self.range
        assert U in self.source
        assert len(U) == 1
        assert len(V) == 1
        assert U_ind is None
        assert V_ind is None
        assert product is None
        # we only work for diagonals blocks atm
        assert self.num_range_blocks == self.num_source_blocks
        assert all([self._blocks[ii][jj] is None if jj != ii else True
                    for jj in np.arange(self.num_source_blocks)
                   for ii in np.arange(self.num_range_blocks)])
        return sum([self._blocks[ii][ii].apply2(V.block(ii, copy=False), U.block(ii, copy=False), pairwise)
                    for ii in np.arange(self.num_range_blocks)])

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        if self.source.dim == 0 and self.range.dim == 0:
            return source.zeros(len(U))
        else:
            assert len(U) == 1  # not implemented yet
            op = NumpyMatrixOperator(bmat([[coo_matrix(self._blocks[ii][jj].assemble(mu)._matrix) if self._blocks[ii][jj] is not None
                                            else coo_matrix((self._range_dims[ii], self._source_dims[jj]))
                                            for jj in np.arange(self.num_source_blocks)]
                                           for ii in np.arange(self.num_range_blocks)]).todense())
            res = op.apply_inverse(U, ind, options)
            return BlockVectorArray(res, self._source_dims)

    def projected(self, source_basis, range_basis=None, product=None, name=None):
        assert product is None  # not implemented yet!
        if isinstance(source_basis, VectorArrayInterface):
            raise Exception('Not implemented yet!')
        elif isinstance(source_basis, list):
            if self.range.dim == 1:
                assert range_basis is None or (isinstance(range_basis, list) and all([len(basis) == 0 for basis in range_basis]))
                assert len(source_basis) == self.num_source_blocks
                assert all([source_basis[jj] in self._blocks[jj].source for jj in np.arange(self.num_source_blocks)])
                return BlockOperator([self._blocks[jj].projected(source_basis[jj], range_basis=None, product=None) for jj in np.arange(self.num_source_blocks)])
            else:
                assert len(source_basis) == self.num_source_blocks
                source_rb = source_basis
                assert all([source_rb[jj] in self.source.subtype[jj] if source_rb[jj] is not None else True for jj in np.arange(self.num_source_blocks)])
                range_rb = range_basis if range_basis is not None else source_basis
                assert len(range_rb) == self.num_range_blocks
                assert all([range_rb[ii] in self.range.subtype[ii] if range_rb[ii] is not None else True for ii in np.arange(self.num_range_blocks)])
                return BlockOperator([[self._blocks[ii][jj].projected(source_rb[jj], range_rb[ii]) if self._blocks[ii][jj] is not None
                                       else None
                                       for jj in np.arange(self.num_source_blocks)]
                                      for ii in np.arange(self.num_range_blocks)],
                                     [len(base) for base in source_rb],
                                     [len(base) for base in range_rb])
        else:
            raise Exception('Invalid source_basis given!')

    def assemble(self, mu=None):
        assert all([isinstance(block, NumpyMatrixBasedOperator) for block in self._blocks])
        if self.range.dim == 1:
            return NumpyMatrixOperator(np.concatenate([block.assemble(mu).as_vector().data for block in self._blocks], axis=1))
        else:
            raise Exception('Not implemented yet!')

    def as_vector(self, mu=None):
        assert self.range.dim == 1
        return NumpyVectorArray(np.concatenate([self._blocks[jj].assemble(mu)._matrix for jj in np.arange(self.num_source_blocks)], axis=1))
