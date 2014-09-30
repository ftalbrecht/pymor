# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number
import numpy as np

from pymor.core.interfaces import abstractmethod, abstractproperty
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray


class BlockVectorArray(VectorArrayInterface):
    pass


def create_block_vector_array_type(vector_arrays_types):
    assert isinstance(vector_arrays_types, list)
    assert all(issubclass(vv, VectorArrayInterface) for vv in vector_arrays_types)
    # for the moment we assume all blocks to be of the same type
    # this makes some stuff easier, e.g. choosing a name for the final class
    assert all(type(vv) == type(vector_arrays_types[0]) for vv in vector_arrays_types)

    class BlockVectorArrayWrapper(BlockVectorArray):

        _block_types = vector_arrays_types

        @classmethod
        def make_array(cls, subtype=None, count=0, reserve=0):
            if not isinstance(subtype, list):
                subtype = [subtype for bb in cls._block_types]
            return BlockVectorArrayWrapper([block_type.make_array(subtp, count=count, reserve=reserve)
                                           for block_type, subtp in izip(cls._block_types, subtype)])

        def __init__(self, blocks, block_sizes=None, copy=False):
            if isinstance(blocks, list):
                # we assume we get a list of compatible vector arrays
                assert block_sizes is None
                assert len(blocks) > 0
                assert all([type(block) == block_type for block, block_type in izip(blocks, self._block_types)])
                self._blocks = [block.copy() for block in blocks] if copy else blocks
            else:
                # we assume we are given a vector array and a list of block sizes
                # we slice the vector into appropriate blocks and create vector arrays
                assert isinstance(blocks, VectorArrayInterface)
                assert block_sizes is not None
                assert isinstance(block_sizes, list)
                assert all(isinstance(block_size, Numer) for block_size in block_sizes)
                assert blocks.dim == sum(block_sizes)
                self._blocks = [block_type(blocks.components(range(sum(block_sizes[:ss]), sum(block_sizes[:(ss + 1)]))))
                                for block_type, ss in izip(self._block_types, np.arange(len(block_sizes)))]

        def block(self, ind, copy=False):
            if isinstance(ind, list):
                assert all(isinstance(ii, Number) for ii in ind)
                return [self._blocks[ii].copy() for ii in ind] if copy else [self._blocks[ii] for ii in ind]
            else:
                assert isinstance(ind, Number)
                return self._blocks[ind].copy() if copy else self._blocks[ind]

        @property
        def subtype(self):
            return [block.subtype for block in self._blocks]

        @property
        def num_blocks(self):
            return len(self.subtype)

        def __len__(self):
            assert all([len(block) == len(self._blocks[0]) for block in self._blocks])
            return len(self._blocks[0])

        @property
        def dim(self):
            return sum([block.dim for block in self._blocks])

        def copy(self, ind=None):
            return BlockVectorArrayWrapper([block.copy(ind) for block in self._blocks], copy=False)

        def append(self, other, o_ind=None, remove_from_other=False):
            raise NotImplementedError

        def remove(self, ind):
            raise NotImplementedError

        def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
            raise NotImplementedError

        def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
            raise NotImplementedError

        def scal(self, alpha, ind=None):
            raise NotImplementedError

        def axpy(self, alpha, x, ind=None, x_ind=None):
            assert x in self.space
            if len(x) > 0:
                for s_block, x_block in zip(self._blocks, x._blocks):
                    s_block.axpy(alpha, x_block, ind, x_ind)

        def dot(self, other, pairwise, ind=None, o_ind=None):
            raise NotImplementedError

        def lincomb(self, coefficients, ind=None):
            raise NotImplementedError

        def l1_norm(self, ind=None):
            raise NotImplementedError

        def l2_norm(self, ind=None):
            assert len(self) == 1
            assert ind == 1 or ind is None
            return np.sqrt(sum([block.l2_norm() for block in self._blocks]))

        def sup_norm(self, ind=None):
            raise NotImplementedError

        def components(self, component_indices, ind=None):
            raise NotImplementedError

        def amax(self, ind=None):
            raise NotImplementedError

        def gramian(self, ind=None):
            raise NotImplementedError

    BlockVectorArrayWrapper.__name__ = 'Block_' + vector_arrays_types[0].__name__

    return BlockVectorArrayWrapper

