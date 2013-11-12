# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.interfaces import abstractmethod, abstractproperty
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray


class BlockVectorArray(VectorArrayInterface):

    @classmethod
    def empty(cls, dim, reserve=0):
        raise Exception('Not yet implemented!')

    @classmethod
    def zeros(cls, dim, count=1):
        raise Exception('Not yet implemented!')

    def __init__(self, blocks, block_sizes=None, copy=False):
        if block_sizes is None:
            assert isinstance(blocks, list)
            assert len(blocks) > 0
            assert all([isinstance(block, VectorArrayInterface) for block in blocks])
            assert all([len(block) == len(blocks[0]) for block in blocks])
            if copy:
                self._blocks = [block.copy() for block in blocks]
            else:
                self._blocks = blocks
        else:
            assert isinstance(block_sizes, list)
            assert all([ss > 0 for ss in block_sizes])
            assert isinstance(blocks, VectorArrayInterface)
            assert blocks.dim == sum(block_sizes)
            self._blocks = [NumpyVectorArray(blocks.components(range(sum(block_sizes[:ss]),
                                                                     sum(block_sizes[:(ss + 1)])))) for ss in np.arange(len(block_sizes))]
        self.type_blocks = [type(block) for block in self._blocks]

    def block(self, ind, copy=False):
        if isinstance(ind, list):
            return [self._blocks[ii].copy() for ii in ind] if copy else [self._blocks[ii] for ii in ind]
        else:
            return self._blocks[ind].copy() if copy else self._blocks[ind]

    @property
    def block_dims(self):
        return [block.dim for block in self._blocks]

    @property
    def num_blocks(self):
        return len(self._blocks)

    def __len__(self):
        assert all([len(block) == len(self._blocks[0]) for block in self._blocks])
        return len(self._blocks[0])

    @property
    def dim(self):
        return sum([block.dim for block in self._blocks])

    def copy(self, ind=None):
        return BlockVectorArray([block.copy(ind) for block in self._blocks], copy=False)

    def append(self, other, o_ind=None, remove_from_other=False):
        raise Exception('Not yet implemented!')

    def remove(self, ind):
        raise Exception('Not yet implemented!')

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        raise Exception('Not yet implemented!')

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        raise Exception('Not yet implemented!')

    def scal(self, alpha, ind=None):
        raise Exception('Not yet implemented!')

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert isinstance(x, BlockVectorArray)
        assert x.num_blocks == self.num_blocks
        assert x.block_dims == self.block_dims
        assert x.type_blocks == self.type_blocks
        if len(x) > 0:
            for s_block, x_block in zip(self._blocks, x._blocks):
                s_block.axpy(alpha, x_block, ind, x_ind)

    def dot(self, other, pairwise, ind=None, o_ind=None):
        raise Exception('Not yet implemented!')

    def lincomb(self, coefficients, ind=None):
        raise Exception('Not yet implemented!')

    def l1_norm(self, ind=None):
        raise Exception('Not yet implemented!')

    def l2_norm(self, ind=None):
        assert len(self) == 1
        assert ind == 1 or ind is None
        return np.sqrt(sum([block.l2_norm() for block in self._blocks]))

    def sup_norm(self, ind=None):
        raise Exception('Not yet implemented!')

    def components(self, component_indices, ind=None):
        raise Exception('Not yet implemented!')

    def amax(self, ind=None):
        raise Exception('Not yet implemented!')

    def gramian(self, ind=None):
        raise Exception('Not yet implemented!')
