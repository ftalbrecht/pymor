# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.operators.interfaces import OperatorInterface
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.la.blockvectorarray import BlockVectorArray
from pymor.operators.basic import NumpyMatrixOperator, NumpyMatrixBasedOperator


class BlockOperator(OperatorInterface):

    def __init__(self, blocks):
        assert isinstance(blocks, list)
        if isinstance(blocks[0], OperatorInterface):
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
        else:
            raise Exception('Not implemented yet!')
        self.type_source = BlockVectorArray
        self.lock()

    def apply(self, U, ind=None, mu=None):
        raise Exception('Not implemented yet!')

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        raise Exception('Not implemented yet!')

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        raise Exception('Not implemented yet!')

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
        assert range_basis is None or self.dim_range > 1
        assert product is None  # not implemented yet!
        if isinstance(source_basis, VectorArrayInterface):
            raise Exception('Not implemented yet!')
        elif isinstance(source_basis, list):
            if self.dim_range == 1:
                assert len(source_basis) == self.num_source_blocks
                assert all([source_basis[ss].dim == self._blocks[ss].dim_source for ss in np.arange(self.num_source_blocks)])
                return BlockOperator([self._blocks[ss].projected(source_basis[ss], range_basis=None, product=None) for ss in np.arange(self.num_source_blocks)])
            else:
                raise Exception('Not implemented yet!')

    def assemble(self, mu=None):
        assert all([isinstance(block, NumpyMatrixBasedOperator) for block in self._blocks])
        if self.dim_range == 1:
            return NumpyMatrixOperator(np.concatenate([block.assemble(mu).as_vector().data for block in self._blocks], axis=1))
        else:
            raise Exception('Not implemented yet!')
