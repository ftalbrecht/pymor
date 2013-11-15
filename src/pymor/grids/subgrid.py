# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import weakref

import numpy as np

import pymor.core as core
from pymor.domaindescriptions import BoundaryType
from pymor.grids.interfaces import AffineGridInterface


class SubGrid(AffineGridInterface):

    reference_element = None

    def __init__(self, grid, entities):
        assert isinstance(grid, AffineGridInterface)
        super(SubGrid, self).__init__()
        self.dim = grid.dim
        self.dim_outer = grid.dim_outer
        self.reference_element = grid.reference_element

        parent_indices = [np.array(np.unique(entities), dtype=np.int32)]
        assert len(parent_indices[0] == len(entities))

        subentities = [np.arange(len(parent_indices[0]), dtype=np.int32).reshape((-1,1))]

        for codim in xrange(1, self.dim + 1):
            SUBE = grid.subentities(0, codim)[parent_indices[0]]
            if np.any(SUBE < 0):
                raise NotImplementedError
            UI, UI_inv = np.unique(SUBE, return_inverse=True)
            subentities.append(np.array(UI_inv.reshape(SUBE.shape), dtype=np.int32))
            parent_indices.append(np.array(UI, dtype=np.int32))

        self.__parent_grid = weakref.ref(grid)
        self.__parent_indices = parent_indices
        self.__subentities = subentities
        embeddings = grid.embeddings(0)
        self.__embeddings = (embeddings[0][parent_indices[0]], embeddings[1][parent_indices[0]])

    @property
    def parent_grid(self):
        return self.__parent_grid()

    def parent_indices(self, codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        return self.__parent_indices[codim]

    def indices_from_parent_indices(self, ind, codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        ind = ind.ravel()
        # TODO Find better implementation of the following
        R = np.argmax(ind[:, np.newaxis] - self.__parent_indices[codim][np.newaxis, :] == 0, axis=1)
        if not np.all(self.__parent_indices[codim][R] == ind):
            raise ValueError('Not all parent indices found')
        return np.array(R, dtype=np.int32)

    def size(self, codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        return len(self.__parent_indices[codim])

    def subentities(self, codim, subentity_codim=None):
        if codim == 0:
            if subentity_codim is None:
                subentity_codim = codim + 1
            assert codim <= subentity_codim <= self.dim, 'Invalid subentity codimension'
            return self.__subentities[subentity_codim]
        else:
            return super(SubGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim):
        if codim == 0:
            return self.__embeddings
        else:
            return super(SubGrid, self).embeddings(codim)

    def test_instances():
        from pymor.grids.rect import RectGrid
        from pymor.grids.tria import TriaGrid
        import random
        import math as m
        grids = [RectGrid((1,1))] #, TriaGrid((1,1)), RectGrid((8,8)), TriaGrid((24,24))]
        rstate = random.getstate()
        subgrids = []
        for g in grids:
            size = g.size(0)
            subgrids.append(SubGrid(g, np.arange(size, dtype=np.int32)))
            if size >= 4:
                subgrids.append(SubGrid(g, np.array(random.sample(xrange(size), int(m.floor(size / 4))))))
            if size >= 2:
                subgrids.append(SubGrid(g, np.array(random.sample(xrange(size), int(m.floor(size / 2))))))
        return subgrids
