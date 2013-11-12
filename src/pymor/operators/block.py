# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, bmat

from pymor.operators.interfaces import OperatorInterface
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.la.blockvectorarray import BlockVectorArray
from pymor.operators.basic import NumpyMatrixOperator, NumpyMatrixBasedOperator


class BlockOperator(OperatorInterface):

    def __init__(self, blocks, source_dims=None, range_dims=None):
        assert isinstance(blocks, list)
        if isinstance(blocks[0], OperatorInterface):
            assert source_dims is None and range_dims is None
            assert all([isinstance(block, OperatorInterface) for block in blocks])
            if blocks[0].dim_range == 1:
                assert all([block.dim_range == 1 for block in blocks])
                self.dim_range = 1
                self._blocks = blocks
                self.num_source_blocks = len(self._blocks)
                self.num_range_blocks = 1
                self.linear = all([block.linear for block in self._blocks])
                self.invert_options = None
                self.type_range = NumpyVectorArray
            else:
                raise Exception('Not implemented yet!')
            self.dim_source = sum([block.dim_source for block in blocks])
            self.build_parameter_type(inherits=self._blocks)
        else:
            assert source_dims is not None and range_dims is not None
            assert isinstance(source_dims, list) and isinstance(range_dims, list)
            self._range_dims = range_dims
            self._source_dims = source_dims
            self.dim_range = sum(self._range_dims)
            self.dim_source = sum(self._source_dims)
            self.num_source_blocks = len(self._source_dims)
            self.num_range_blocks = len(self._range_dims)
            assert len(blocks) == len(self._range_dims)
            assert all([len(block_row) == len(self._source_dims) for block_row in blocks])
            assert all([all([isinstance(blocks[ii][jj], OperatorInterface) if blocks[ii][jj] is not None else True for jj in np.arange(self.num_source_blocks)]) for ii in np.arange(self.num_range_blocks)])
            assert all([all([blocks[ii][jj].dim_source == self._source_dims[jj] if blocks[ii][jj] is not None else True for jj in np.arange(self.num_source_blocks)]) for ii in np.arange(self.num_range_blocks)])
            assert all([all([blocks[ii][jj].dim_range == self._range_dims[ii] if blocks[ii][jj] is not None else True for jj in np.arange(self.num_source_blocks)]) for ii in np.arange(self.num_range_blocks)])
            self._blocks = blocks
            self.build_parameter_type(inherits=[block for block in block_row for block_row in blocks])
            self.linear = all([all([block.linear if block is not None else True for block in blocks_row]) for blocks_row in self._blocks])
            self.invert_options = None
            self.type_range = BlockVectorArray
        self.type_source = BlockVectorArray

    def apply(self, U, ind=None, mu=None):
        raise Exception('Not implemented yet!')

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        assert isinstance(V, BlockVectorArray)
        assert isinstance(U, BlockVectorArray)
        assert V.num_blocks == self.num_range_blocks
        assert V.block_dims == self._range_dims
        assert U.num_blocks == self.num_source_blocks
        assert U.block_dims == self._source_dims
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
        if self.dim_source == 0 and self.dim_range == 0:
            return BlockVectorArray([NumpyVectorArray(np.zeros((0, 0), dtype=float)) for jj in np.arange(self.num_range_blocks)])
        else:
            op = NumpyMatrixOperator(bmat([[coo_matrix(self._blocks[ii][jj].assemble(mu)._matrix) if self._blocks[ii][jj] is not None
                                            else coo_matrix((self._range_dims[ii], self._source_dims[jj]))
                                            for jj in np.arange(self.num_source_blocks)]
                                           for ii in np.arange(self.num_range_blocks)]).todense())
        res = op.apply_inverse(U, ind, options)
        assert len(res) == 1  # not implemented yet
        return BlockVectorArray(res, self._source_dims)

    @staticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        raise Exception('Not implemented yet!')

    def __add__(self, other):
        raise Exception('Not implemented yet!')

    def __radd__(self, other):
        raise Exception('Not implemented yet!')

    def __mul__(self, other):
        raise Exception('Not implemented yet!')

    def projected(self, source_basis, range_basis=None, product=None, name=None):
        assert product is None  # not implemented yet!
        if isinstance(source_basis, VectorArrayInterface):
            raise Exception('Not implemented yet!')
        elif isinstance(source_basis, list):
            if self.dim_range == 1:
                assert range_basis is None or (isinstance(range_basis, list) and all([len(basis) == 0 for basis in range_basis]))
                assert len(source_basis) == self.num_source_blocks
                assert all([source_basis[jj].dim == self._blocks[jj].dim_source for jj in np.arange(self.num_source_blocks)])
                return BlockOperator([self._blocks[jj].projected(source_basis[jj], range_basis=None, product=None) for jj in np.arange(self.num_source_blocks)])
            else:
                assert len(source_basis) == self.num_source_blocks
                source_rb = source_basis
                assert all([source_rb[jj].dim == self._source_dims[jj] if source_rb[jj] is not None else True for jj in np.arange(self.num_source_blocks)])
                range_rb = range_basis if range_basis is not None else source_basis
                assert len(range_rb) == self.num_range_blocks
                assert all([range_rb[ii].dim == self._range_dims[ii] if range_rb[ii] is not None else True for ii in np.arange(self.num_range_blocks)])
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
        if self.dim_range == 1:
            return NumpyMatrixOperator(np.concatenate([block.assemble(mu).as_vector().data for block in self._blocks], axis=1))
        else:
            raise Exception('Not implemented yet!')

    def as_vector(self, mu=None):
        assert self.dim_range == 1
        return NumpyVectorArray(np.concatenate([self._blocks[jj].assemble(mu)._matrix for jj in np.arange(self.num_source_blocks)], axis=1))
