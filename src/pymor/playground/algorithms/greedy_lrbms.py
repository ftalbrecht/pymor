# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import time
from itertools import izip
import numpy as np

from pymor.algorithms.basisextension import trivial_basis_extension
from pymor.core import getLogger
from pymor.core.exceptions import ExtensionError


def greedy_lrbms(discretization, reductor, samples, initial_basis=None, use_estimator=True, error_norm=None,
                 extension_algorithm=trivial_basis_extension, target_error=None, max_extensions=None):
    '''Greedy basis generation algorithm in the LRBMS context.

    This algorithm generates a reduced basis by iteratively adding the
    worst approximated solution snapshot for a given training set to the
    reduced basis. The approximation error is computed either by directly
    comparing the reduced solution to the detailed solution or by using
    an error estimator (`use_estimator == True`). The reduction and basis
    extension steps are performed by calling the methods provided by the
    `reductor` and `extension_algorithm` arguments.

    Parameters
    ----------
    discretization
        The discretization to reduce.
    reductor
        Reductor for reducing the given discretization. This has to be a
        function of the form `reductor(discretization, basis, extends=None)`.
        If your reductor takes more arguments, use, e.g., functools.partial.
        The method has to return a tuple
        `(reduced_discretization, reconstructor, reduction_data)`.
        In case the last basis extension was `hierarchic` (see
        `extension_algorithm`), the extends argument is set to
        `(last_reduced_discretization, last_reconstructor, last_reduction_data)`
        which can be used by the reductor to speed up the reduction
        process. For an example see
        :func:`pymor.reductors.linear.reduce_stationary_affine_linear`.
    samples
        The set of parameter samples on which to perform the greedy search.
    initial_basis
        The initial reduced basis with which the algorithm starts. If `None`,
        an empty basis is used as initial_basis.
    use_estimator
        If `True`, use `reduced_discretization.estimate()` to estimate the
        errors on the sample set. Otherwise a detailed simulation is
        performed to calculate the error.
    error_norm
        If `use_estimator == False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    extension_algorithm
        The extension algorithm to be used to extend the current reduced
        basis with the maximum error snapshot. This has to be a function
        of the form `extension_algorithm(old_basis, new_vector)`, which
        returns a tuple `(new_basis, extension_data)`, where
        `extension_data` is a dict at least containing the key
        `hierarchic`. `hierarchic` is set to `True` if `new_basis`
        contains `old_basis` as its first vectors.
    target_error
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the sample set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.

    Returns
    -------
    Dict with the following fields:
        'basis'
            The reduced basis.
        'reduced_discretization'
            The reduced discretization obtained for the computed basis.
        'reconstructor'
            Reconstructor for `reduced_discretization`.
        'max_err'
            Last estimated maximum error on the sample set.
        'max_err_mu'
            The parameter that corresponds to `max_err`.
        'max_errs'
            Sequence of maximum errors during the greedy run.
        'max_errs_mu'
            The parameters corresponding to `max_errs`.
    '''

    logger = getLogger('pymor.algorithms.greedy.greedy')
    samples = list(samples)
    assert isinstance(initial_basis, list)
    num_subdomains = len(initial_basis)
    if isinstance(extension_algorithm, list):
        assert len(extension_algorithm) == num_subdomains
    else:
        extension_algorithm = [extension_algorithm for ss in np.arange(num_subdomains)]
    basis = initial_basis
    logger.info('Started greedy search on {} samples, {} spatial subdomains'.format(len(samples), num_subdomains))

    tic = time.time()
    extensions = 0
    max_errs = []
    max_err_mus = []
    hierarchic = False

    while True:
        logger.info('Reducing ...')
        rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
            else reductor(discretization, basis, extends=(rd, rc, reduction_data))

        logger.info('Estimating errors ...')
        if use_estimator:
            errors = [rd.estimate(rd.solve(mu), mu) for mu in samples]
        elif error_norm is not None:
            errors = [error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))) for mu in samples]
        else:
            errors = [(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))).l2_norm() for mu in samples]

        # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
        # if necessary
        errors = map(lambda x: x[0] if hasattr(x, '__len__') else x, errors)

        max_err, max_err_mu = max(((err, mu) for err, mu in izip(errors, samples)), key=lambda t: t[0])
        max_errs.append(max_err)
        max_err_mus.append(max_err_mu)
        logger.info('Maximum error after {} extensions: {} (mu = {})'.format(extensions, max_err, max_err_mu))

        if target_error is not None and max_err <= target_error:
            logger.info('Reached maximal error on snapshots of {} <= {}'.format(max_err, target_error))
            break

        logger.info('Extending with snapshot for mu = {}'.format(max_err_mu))
        U = discretization.solve(max_err_mu)
        assert U.num_blocks == num_subdomains
        local_bases_extended = 0
        for ss in np.arange(num_subdomains):
            try:
                basis[ss], _  = extension_algorithm[ss](basis[ss], U.block(ss))
                local_bases_extended += 1
            except ExtensionError:
                logger.info('Extension failed on subdomain {}.'.format(ss))
            # if not 'hierarchic' in extension_data:
            #     logger.warn('Extension algorithm does not report if extension was hierarchic. Assuming it was\'nt ..')
            hierarchic = False
            # else:
            #     hierarchic = extension_data['hierarchic']
        extensions += 1

        if local_bases_extended == 0:
            logger.info('Extension failed on all subdomains. Stopping now.')
            break

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info('Maximal number of {} extensions reached.'.format(max_extensions))
            logger.info('Reducing once more ...')
            rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
                else reductor(discretization, basis, extends=(rd, rc, reduction_data))
            break

    tictoc = time.time() - tic
    logger.info('Greedy search took {} seconds'.format(tictoc))
    return {'basis': basis, 'reduced_discretization': rd, 'reconstructor': rc, 'max_err': max_err,
            'max_err_mu': max_err_mu, 'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
            'time': tictoc, 'reduction_data': reduction_data}