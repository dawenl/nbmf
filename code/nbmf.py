"""

Negative-binomial matrix factorization with Batch and Stochastic inference

CREATED: 2014-11-18 20:50:49 by Dawen Liang <dliang@ee.columbia.edu>

"""


import sys
import numpy as np
from scipy import special

from math import log
from sklearn.base import BaseEstimator, TransformerMixin


class NegBinomMF(BaseEstimator, TransformerMixin):
    ''' Negative-binomial matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,
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
                                                        size=X.shape
                                                        ).astype(np.float32)
        self.rho_lam = 1. / X * self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness, size=X.shape
                            ).astype(np.float32)
        #self.rho_lam = self.smoothness * np.random.gamma(self.smoothness,
        #                                                 1. / self.smoothness,
        #                                                 size=X.shape
        #                                                 ).astype(np.float32)
        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _init_users(self, n_users):
        # variational parameters for theta
        self.rho_t = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(self.n_components, n_users)
                                             ).astype(np.float32)
        self.tau_t = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(self.n_components, n_users)
                                             ).astype(np.float32)
        self.Et, self.Etinv = comp_gig_expectations(self.a,
                                                    self.rho_t,
                                                    self.tau_t)
        self.Etinvinv = 1. / self.Etinv

    def _init_items(self, n_items):
        # variational parameters for beta

        self.rho_b = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(n_items, self.n_components)
                                             ).astype(np.float32)
        self.tau_b = 10000 * np.random.gamma(self.smoothness,
                                             1. / self.smoothness,
                                             size=(n_items, self.n_components)
                                             ).astype(np.float32)
        self.Eb, self.Ebinv = comp_gig_expectations(self.c,
                                                    self.rho_b,
                                                    self.tau_b)
        self.Ebinvinv = 1. / self.Ebinv

    def fit(self, X):
        '''Fit the model to the data in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
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

    #def transform(self, X, attr=None):
    #    '''Encode the data as a linear combination of the latent components.
    #    Parameters
    #    ----------
    #    X : array-like, shape (n_samples, n_feats)
    #    attr: string
    #        The name of attribute, default 'Eb'. Can be changed to Elogb to
    #        obtain E_q[log beta] as transformed data.
    #    Returns
    #    -------
    #    X_new : array-like, shape(n_samples, n_filters)
    #        Transformed data, as specified by attr.
    #    '''
    #
    #    if not hasattr(self, 'Eb'):
    #        raise ValueError('There are no pre-trained components.')
    #    n_samples, n_feats = X.shape
    #    if n_feats != self.Eb.shape[1]:
    #        raise ValueError('The dimension of the transformed data '
    #                         'does not match with the existing components.')
    #    if attr is None:
    #        attr = 'Et'
    #    self._init_weights(n_samples)
    #    self._update(X, update_beta=False)
    #    return getattr(self, attr)

    def _update(self, X):
        # alternating between update users and items
        old_bd = -np.inf
        for i in xrange(self.max_iter):
            for _ in xrange(10):
                self._update_users(X)
                self._update_items(X)
            self._update_aux(X)
            bound = self._bound(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('ITERATION: %d\tObjective: %.2f\t'
                      'Old objective: %.2f\t'
                      'Improvement: %.5f' % (i, bound, old_bd, improvement))
                sys.stdout.flush()
            #if improvement < self.tol:
            #    break
            old_bd = bound
        #if self.verbose:
        #    sys.stdout.write('\n')
        pass

    def _update_users(self, X):
        EXinv = 1. / self.Eb.dot(self.Et)
        laminvXsq = self.Elam / (self.Ebinvinv.dot(self.Etinvinv))**2
        self.rho_t = self.b + self.r * self.Eb.T.dot(EXinv)
        self.tau_t = self.r * self.Etinvinv**2 * self.Ebinvinv.T.dot(laminvXsq)
        self.tau_t[self.tau_t < 1e-100] = 0
        self.Et, self.Etinv = comp_gig_expectations(self.a,
                                                    self.rho_t,
                                                    self.tau_t)
        self.Etinvinv = 1. / self.Etinv

    def _update_items(self, X):
        EXinv = 1. / self.Eb.dot(self.Et)
        laminvXsq = self.Elam / (self.Ebinvinv.dot(self.Etinvinv))**2
        self.rho_b = self.d + self.r * EXinv.dot(self.Et.T)
        self.tau_b = self.r * self.Ebinvinv**2 * laminvXsq.dot(self.Etinvinv.T)
        self.tau_b[self.tau_b < 1e-100] = 0
        self.Eb, self.Ebinv = comp_gig_expectations(self.c,
                                                    self.rho_b,
                                                    self.tau_b)
        self.Ebinvinv = 1. / self.Ebinv

    def _update_aux(self, X):
        self.nu_lam = self.r + X
        self.rho_lam = 1 + self.r / self.Ebinvinv.dot(self.Etinvinv)
        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _bound(self, X):
        bound = np.sum(special.gammaln(self.nu_lam) -
                       self.nu_lam * np.log(self.rho_lam))
        bound -= self.r * (np.log(self.Eb.dot(self.Et))).sum()
        bound += gig_gamma_term(self.Et, self.Etinv, self.rho_t, self.tau_t,
                                self.a, self.b)
        bound += gig_gamma_term(self.Eb, self.Ebinv, self.rho_b, self.tau_b,
                                self.c, self.d)
        return bound


class OnlineNegBinomMF(NegBinomMF):
    ''' Negtive-binomial matrix factorization with stochastic inference '''
    def __init__(self, n_components=100, batch_size=10, n_pass=10,
                 max_iter=100, tol=0.0005, shuffle=True, smoothness=100,
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
        self.max_iter = max_iter
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
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def fit(self, X, est_total=None):
        '''Fit the model to the data in X. X has to be loaded into memory.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.
        est_total : int
            The estimated size of the entire data. Could be larger than the
            actual size.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_samples, n_feats = X.shape
        if est_total is None:
            self._scale = float(n_samples) / self.batch_size
        else:
            self._scale = float(est_total) / self.batch_size
        self._init_components(n_feats)
        self.bound = list()
        for count in xrange(self.n_pass):
            if self.verbose:
                print 'Iteration %d: passing through the data...' % count
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            X_shuffled = X[indices]
            for (i, istart) in enumerate(xrange(0, n_samples,
                                                self.batch_size), 1):
                print '\tMinibatch %d:' % i
                iend = min(istart + self.batch_size, n_samples)
                self.set_learning_rate(iter=i)
                mini_batch = X_shuffled[istart: iend]
                self.partial_fit(mini_batch)
                self.bound.append(self._stoch_bound(mini_batch))
        return self

    def partial_fit(self, X):
        '''Fit the data in X as a mini-batch and update the parameter by taking
        a natural gradient step. Could be invoked from a high-level out-of-core
        wrapper.
        Parameters
        ----------
        X : array-like, shape (batch_size, n_feats)
            Mini-batch data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.transform(X)
        # take a (natural) gradient step
        ratio = X / self._xexplog()
        self.gamma_b = (1 - self.rho) * self.gamma_b + self.rho * \
            (self.b + self._scale * np.exp(self.Elogb) *
             np.dot(np.exp(self.Elogt).T, ratio))
        self.rho_b = (1 - self.rho) * self.rho_b + self.rho * \
            (self.b + self._scale * np.sum(self.Et, axis=0, keepdims=True).T)
        self.Eb, self.Elogb = comp_gamma_expectations(self.gamma_b, self.rho_b)
        return self

    def set_learning_rate(self, iter=None, rho=None):
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
        if rho is not None:
            self.rho = rho
        elif iter is not None:
            self.rho = (iter + self.t0)**(-self.kappa)
        else:
            raise ValueError('invalid learning rate.')
        return self

    def _stoch_bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += gamma_term(self.a, self.a * self.c, self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound *= self._scale
        bound += gamma_term(self.b, self.b, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound


def comp_gamma_expectations(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def comp_gamma_entropy(alpha, beta):
    ''' Compute the entropy of a r.v. theta ~ Gamma(alpha, beta)
    '''
    return (alpha - np.log(beta) + special.gammaln(alpha) +
            (1 - alpha) * special.psi(alpha))


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


def gamma_term(Ex, Elogx, shape, rate, a, b):
    ''' Compute E_q[log p(x)] - E_q[log q(x)] where:
        p(x) = Gamma(a, b), q(x) = Gamma(shape, rate)
    '''
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))


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
