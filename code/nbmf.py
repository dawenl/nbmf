"""

Negative-binomial matrix factorization with Batch and Stochastic inference

CREATED: 2014-11-18 20:50:49 by Dawen Liang <dliang@ee.columbia.edu>

"""


import sys
import numpy as np
from scipy import sparse, special

from math import log
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_NUM_ITER = 1
EPS = np.spacing(1)


class NegBinomMF(BaseEstimator, TransformerMixin):
    ''' Negative-binomial matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, max_inner_iter=20,
                 tol=0.0005, smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        ''' Negative-binomial matrix factorization
        Arguments
        ---------
        n_components : int
            Number of latent components
        max_iter : int
            Maximal number of iterations to perform
        tol : float
            The threshold on the increase of the objective to stop the
            iteration
        smoothness : int
            Smoothness on the initialization variational parameters
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''

        self.n_components = n_components
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

        self.r = float(kwargs.get('r', 2.0))

    def _init_aux(self, X):
        # variational parameters for lambda
        self.nu_lam = self.smoothness * np.random.gamma(self.smoothness,
                                                        1. / self.smoothness,
                                                        size=X.shape)
        #self.rho_lam = 1. / (X + EPS) * self.smoothness * \
        #    np.random.gamma(self.smoothness, 1. / self.smoothness,
        #                    size=X.shape)
        self.rho_lam = self.smoothness * np.random.gamma(self.smoothness,
                                                         1. / self.smoothness,
                                                         size=X.shape)
        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _init_users(self, n_users):
        # variational parameters for theta
        self.rho_t = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(self.n_components, n_users))
        self.tau_t = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(self.n_components, n_users))
        self.Et, self.Etinv = comp_gig_expectations(self.a,
                                                    self.rho_t,
                                                    self.tau_t)
        self.Etinvinv = 1. / self.Etinv

    def _init_items(self, n_items):
        # variational parameters for beta
        self.rho_b = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(n_items, self.n_components))
        self.tau_b = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(n_items, self.n_components))
        self.Eb, self.Ebinv = comp_gig_expectations(self.c,
                                                    self.rho_b,
                                                    self.tau_b)
        self.Ebinvinv = 1. / self.Ebinv

    def fit(self, X):
        '''Fit the model to the data in X.
        Parameters
        ----------
        X : array-like, shape (n_items, n_users)
            Training data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_items, n_users = X.shape
        self._init_aux(X)
        self._init_items(n_items)
        self._init_users(n_users)
        self._update(X)
        return self

    def transform(self, X, idx=None, attr=None):
        # sanity check
        if not hasattr(self, 'Et'):
            raise ValueError('There are no pre-trained user latent factors.')
        n_items, n_users = X.shape
        if n_users != self.Et.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing factors.')
        if attr is None:
            attr = 'Eb'

        if sparse.isspmatrix(X):
            X = X.toarray()
        self._init_aux(X)
        if idx is None:
            self._init_items(n_items)
        self._update(X, update_users=False, idx=idx)
        return getattr(self, attr)

    def _update(self, X, update_users=True, idx=None):
        # alternating between updating users and items
        old_bd = -np.inf
        for i in xrange(self.max_iter):
            old_inner_bd = -np.inf
            for j in xrange(self.max_inner_iter):
                if update_users:
                    self._update_users()
                self._update_items(idx=idx)
                inner_bound = self._bound(X, aux_updated=False, idx=idx)
                improvement = (inner_bound - old_inner_bd) / abs(old_inner_bd)
                if self.verbose:
                    sys.stdout.write('\r\tInner TERATION: %d\tObjective: %.2f'
                                     '\tOld objective: %.2f\t'
                                     'Improvement: %.5f' % (j, inner_bound,
                                                            old_inner_bd,
                                                            improvement))
                    sys.stdout.flush()
                old_inner_bd = inner_bound
            if self.verbose:
                sys.stdout.write('\n')

            self._update_aux(X, idx=idx)
            bound = self._bound(X, idx=idx)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('ITERATION: %d\tObjective: %.2f\t'
                      'Old objective: %.2f\t'
                      'Improvement: %.5f' % (i, bound, old_bd, improvement))
                sys.stdout.flush()
            if improvement < self.tol and improvement > 10:
                break
            old_bd = bound
        pass

    def _update_users(self):
        EXinv = 1. / self.Eb.dot(self.Et)
        laminvXsq = self.Elam / (self.Ebinvinv.dot(self.Etinvinv))**2
        self.rho_t = self.b + self.r * self.Eb.T.dot(EXinv)
        self.tau_t = self.r * self.Etinvinv**2 * self.Ebinvinv.T.dot(laminvXsq)
        self.tau_t[self.tau_t < 1e-100] = 0
        self.Et, self.Etinv = comp_gig_expectations(self.a,
                                                    self.rho_t,
                                                    self.tau_t)
        self.Etinvinv = 1. / self.Etinv

    def _update_items(self, idx=None):
        if idx is None:
            idx = slice(None, None)
        EXinv = 1. / self.Eb[idx].dot(self.Et)
        laminvXsq = self.Elam / (self.Ebinvinv[idx].dot(self.Etinvinv))**2
        rho_b = self.d + self.r * EXinv.dot(self.Et.T)
        tau_b = self.r * self.Ebinvinv[idx]**2 * laminvXsq.dot(self.Etinvinv.T)
        tau_b[tau_b < 1e-100] = 0
        self.rho_b[idx], self.tau_b[idx] = rho_b, tau_b
        self.Eb[idx], self.Ebinv[idx] = comp_gig_expectations(self.c,
                                                              rho_b,
                                                              tau_b)
        self.Ebinvinv[idx] = 1. / self.Ebinv[idx]

    def _update_aux(self, X, idx=None):
        if idx is None:
            idx = slice(None, None)
        self.nu_lam = self.r + X
        self.rho_lam = 1 + self.r / self.Ebinvinv[idx].dot(self.Etinvinv)
        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _bound(self, X, aux_updated=True, idx=None):
        if idx is None:
            idx = slice(None, None)
        # E_q [log p(y, lambda | theta, beta) - log q(lambda)]
        bound = np.sum(special.gammaln(self.nu_lam) -
                       self.nu_lam * np.log(self.rho_lam))
        bound -= self.r * (np.log(self.Eb[idx].dot(self.Et))).sum()
        if not aux_updated:
            # if we don't update aux, then this term is not equal 0
            bound += np.sum((self.rho_lam - self.r /
                             self.Ebinvinv[idx].dot(self.Etinvinv) - 1)
                            * self.Elam)
        # E_q [log p(theta) - log q(theta)]
        bound += gig_gamma_term(self.Et, self.Etinv, self.rho_t, self.tau_t,
                                self.a, self.b)
        # E_q [log p(beta) - log q(beta)]
        bound += gig_gamma_term(self.Eb[idx], self.Ebinv[idx], self.rho_b[idx],
                                self.tau_b[idx], self.c, self.d)
        return bound


class OnlineNegBinomMF(NegBinomMF):
    ''' Negtive-binomial matrix factorization with stochastic inference '''
    def __init__(self, n_components=100, batch_size=10, n_pass=10,
                 max_inner_iter=20, tol=0.0005, shuffle=True, smoothness=100,
                 random_state=None, verbose=False,
                 **kwargs):
        ''' Negative-binomial matrix factorization
        Arguments
        ---------
        n_components : int
            Number of latent components
        batch_size : int
            The size of mini-batch
        n_pass : int
            The number of passes through the entire data
        max_iter : int
            Maximal number of iterations to perform for a single mini-batch
        tol : float
            The threshold on the increase of the objective to stop the
            iteration
        shuffle : bool
            Whether to shuffle the data or not
        smoothness : int
            Smoothness on the initialization variational parameters
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters and learning rate
        '''

        self.n_components = n_components
        self.batch_size = batch_size
        self.n_pass = n_pass
        self.max_iter = DEFAULT_NUM_ITER
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.shuffle = shuffle
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

        self.r = float(kwargs.get('r', 2.0))

        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.5))

    def fit(self, X, est_total=None):
        '''Fit the model to the data in X. X can be a scipy.sparse.csr_matrix
        Parameters
        ----------
        X : array-like, shape (n_items, n_users)
            Training data.
        est_total : int
            The estimated size of the entire data. Could be larger than the
            actual size.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_items, n_users = X.shape
        if est_total is None:
            self._scale = float(n_items) / self.batch_size
        else:
            self._scale = float(est_total) / self.batch_size

        self._init_items(n_items)
        self._init_users(n_users)

        self.bound = list()
        for count in xrange(self.n_pass):
            if self.verbose:
                print 'Iteration %d: passing through the data...' % count
            indices = np.arange(n_items)
            if self.shuffle:
                np.random.shuffle(indices)
            for (i, istart) in enumerate(xrange(0, n_items, self.batch_size)):
                print '\tMinibatch %d:' % i
                idx = indices[istart: istart + self.batch_size]
                mini_batch = X[idx]
                self.set_learning_rate(mini_batch, iter=i)
                self.partial_fit(mini_batch, idx=idx)
                self.bound.append(self._stoch_bound(mini_batch, idx=idx))
        return self

    def partial_fit(self, X, idx=None):
        '''Fit the data in X as a mini-batch and update the parameter by taking
        a natural gradient step.
        Parameters
        ----------
        X : array-like, shape (batch_size, n_feats)
            Mini-batch data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.transform(X, idx=idx)
        # take a (natural) gradient step
        EXinv = 1. / self.Eb[idx].dot(self.Et)
        laminvXsq = self.Elam / (self.Ebinvinv[idx].dot(self.Etinvinv))**2
        self.rho_t = (1 - self.rho) * self.rho_t + self.rho * \
            (self.b + self._scale * self.r * self.Eb[idx].T.dot(EXinv))
        self.tau_t = (1 - self.rho) * self.tau_t + self.rho * \
            self._scale * self.r * self.Etinvinv**2 * \
            self.Ebinvinv[idx].T.dot(laminvXsq)
        self.tau_t[self.tau_t < 1e-100] = 0

        self.Et, self.Etinv = comp_gig_expectations(self.a,
                                                    self.rho_t,
                                                    self.tau_t)
        return self

    def set_learning_rate(self, X, iter=None, rho=None):
        '''Set the learning rate for the gradient step
        Parameters
        ----------
        iter : int
            The current iteration, used to compute a Robbins-Monro type
            learning rate
        rho : float
            Directly specify the learning rate. Will override the one computed
            from the current iteration.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        #TODO per-user learning rate
        if rho is not None:
            self.rho = rho
        elif iter is not None:
            self.rho = (iter + self.t0)**(-self.kappa)
        else:
            raise ValueError('invalid learning rate.')
        print '\t\tLearning rate = %.3f' % self.rho
        return self

    def _stoch_bound(self, X, idx=None):
        if idx is None:
            idx = slice(None, None)
        # E_q [log p(y, lambda | theta, beta) - log q(lambda)]
        bound = np.sum(special.gammaln(self.nu_lam) -
                       self.nu_lam * np.log(self.rho_lam))
        bound -= self.r * (np.log(self.Eb[idx].dot(self.Et))).sum()
        # E_q [log p(beta) - log q(beta)]
        bound += gig_gamma_term(self.Eb[idx], self.Ebinv[idx], self.rho_b[idx],
                                self.tau_b[idx], self.c, self.d)
        # scale the objective to "pretent" we have the full dataset
        bound *= self._scale
        # E_q [log p(theta) - log q(theta)]
        bound += gig_gamma_term(self.Et, self.Etinv, self.rho_t, self.tau_t,
                                self.a, self.b)

        return bound


def comp_gamma_expectations(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def comp_gig_expectations(alpha, beta, gamma):
    alpha = alpha * np.ones_like(beta)

    Ex, Exinv = np.zeros_like(beta), np.zeros_like(beta)

    # For very small values of gamma and positive values of alpha, the GIG
    # distribution becomes a gamma distribution, and its expectations are both
    # cheaper and more stable to compute that way.
    gig_inds = (gamma > 1e-200)
    gam_inds = (gamma <= 1e-200)

    if np.any(alpha[gam_inds] < 0):
        raise ValueError("problem with arguments.")

    # Compute expectations for GIG distribution.
    sqrt_beta = np.sqrt(beta[gig_inds])
    sqrt_gamma = np.sqrt(gamma[gig_inds])
    # Note that we're using the *scaled* version here, since we're just
    # computing ratios and it's more stable.
    bessel_alpha_minus = special.kve(alpha[gig_inds] - 1, 2 * sqrt_beta *
                                     sqrt_gamma)
    bessel_alpha = special.kve(alpha[gig_inds], 2 * sqrt_beta * sqrt_gamma)
    bessel_alpha_plus = special.kve(alpha[gig_inds] + 1, 2 * sqrt_beta *
                                    sqrt_gamma)
    sqrt_ratio = sqrt_gamma / sqrt_beta

    Ex[gig_inds] = bessel_alpha_plus * sqrt_ratio / bessel_alpha
    Exinv[gig_inds] = bessel_alpha_minus / (sqrt_ratio * bessel_alpha)

    # Compute expectations for gamma distribution where we can get away with
    # it.
    Ex[gam_inds] = alpha[gam_inds] / beta[gam_inds]
    Exinv[gam_inds] = beta[gam_inds] / (alpha[gam_inds] - 1)
    Exinv[Exinv < 0] = np.inf

    return (Ex, Exinv)


def gig_gamma_term(Ex, Exinv, rho, tau, a, b):
    ''' Compute E_q[log p(x)] - E_q[log q(x)] where:
        p(x) = Gamma(a, b), q(x) = GIG(a, rho, tau)
    '''
    score = 0
    cut_off = 1e-200
    zero_tau = (tau <= cut_off)
    non_zero_tau = (tau > cut_off)

    score = score - np.sum((b - rho) * Ex)
    score = score - np.sum(non_zero_tau) * log(.5)
    score = score + np.sum(tau[non_zero_tau] * Exinv[non_zero_tau])
    score = score + Ex.size * (a * log(b) - special.gammaln(a))
    score = score - .5 * a * np.sum(np.log(rho[non_zero_tau]) -
                                    np.log(tau[non_zero_tau]))
    # It's numerically safer to use scaled version of besselk
    score = score + np.sum(np.log(special.kve(
        a, 2 * np.sqrt(rho[non_zero_tau] * tau[non_zero_tau]))) -
        2 * np.sqrt(rho[non_zero_tau] * tau[non_zero_tau]))
    score = score + np.sum(-a * np.log(rho[zero_tau]) + special.gammaln(a))
    return score
