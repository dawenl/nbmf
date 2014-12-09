"""

Negative-binomial matrix factorization with Batch and Stochastic inference

CREATED: 2014-11-18 20:50:49 by Dawen Liang <dliang@ee.columbia.edu>

"""


import sys
import numpy as np
from scipy import sparse, special
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_NUM_ITER = 1
DEFAULT_MIN_ITER = 10
EPS = np.spacing(1)


class NegBinomMF(BaseEstimator, TransformerMixin):
    ''' Negative-binomial matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, max_inner_iter=20,
                 tol=0.0001, smoothness=100, init_flat_lam=True,
                 random_state=None, verbose=False,
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
        self.init_flat_lam = init_flat_lam
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

        self.r = float(kwargs.get('r', 1.0))

    def _init_aux(self, X):
        # variational parameters for lambda
        self.nu_lam = self.smoothness * np.random.gamma(self.smoothness,
                                                        1. / self.smoothness,
                                                        size=X.shape
                                                        ).astype(np.float32)
        if self.init_flat_lam:
            self.rho_lam = self.smoothness * \
                np.random.gamma(self.smoothness, 1. / self.smoothness,
                                size=X.shape).astype(np.float32)
        else:
            self.rho_lam = 1. / (X + EPS) * self.smoothness * \
                np.random.gamma(self.smoothness, 1. / self.smoothness,
                                size=X.shape).astype(np.float32)

        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _init_users(self, n_users):
        # variational parameters for theta
        self.nu_t = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(self.n_components, n_users)
                            ).astype(np.float32)
        self.rho_t = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(self.n_components, n_users)
                            ).astype(np.float32)
        self.Et, self.Elogt = comp_gamma_expectations(self.nu_t, self.rho_t)

    def _init_items(self, n_items):
        # variational parameters for beta
        self.nu_b = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_items, self.n_components)
                            ).astype(np.float32)
        self.rho_b = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_items, self.n_components)
                            ).astype(np.float32)
        self.Eb, self.Elogb = comp_gamma_expectations(self.nu_b, self.rho_b)

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
                    self._update_users(X)
                self._update_items(X, idx=idx)
                inner_bound = self._bound(X, idx=idx)
                improvement = (inner_bound - old_inner_bd) / abs(old_inner_bd)
                if self.verbose:
                    sys.stdout.write('\r\tInner TERATION: %d\tObjective: %.2f'
                                     '\tOld objective: %.2f\t'
                                     'Improvement: %.5f' % (j, inner_bound,
                                                            old_inner_bd,
                                                            improvement))
                    sys.stdout.flush()
                if improvement < self.tol and j >= DEFAULT_MIN_ITER:
                    break
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
            if improvement < self.tol and i >= DEFAULT_MIN_ITER:
                break
            old_bd = bound
        pass

    def _update_users(self, X):
        ratio = X / self._xexplog()
        self.nu_t = self.a + np.exp(self.Elogt) * np.exp(self.Elogb.T).dot(ratio)
        self.rho_t = self.b + self.Eb.T.dot(self.Elam)
        self.Et, self.Elogt = comp_gamma_expectations(self.nu_t, self.rho_t)

    def _update_items(self, X, idx=None):
        if idx is None:
            idx = slice(None, None)
        ratio = X / self._xexplog(idx=idx)
        nu_b = self.c + np.exp(self.Elogb[idx]) * ratio.dot(np.exp(self.Elogt.T))
        rho_b = self.d + self.Elam.dot(self.Et.T)
        self.nu_b[idx], self.rho_b[idx] = nu_b, rho_b
        self.Eb[idx], self.Elogb[idx] = comp_gamma_expectations(nu_b, rho_b)

    def _update_aux(self, X, idx=None):
        if idx is None:
            idx = slice(None, None)
        self.nu_lam = self.r + X
        self.rho_lam = self.r + self.Eb[idx].dot(self.Et)
        self.Elam, self.Eloglam = comp_gamma_expectations(self.nu_lam,
                                                          self.rho_lam)

    def _xexplog(self, idx=None):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        if idx is None:
            idx = slice(None, None)
        return np.exp(self.Elogb[idx]).dot(np.exp(self.Elogt))

    def _bound(self, X, idx=None):
        if idx is None:
            idx = slice(None, None)
        # E_q [log p(y, lambda | theta, beta) - log q(lambda)]
        bound = np.sum(X * np.log(self._xexplog(idx=idx)))
        bound += gamma_term(self.Elam, self.Eloglam, self.nu_lam, self.rho_lam,
                            self.r + X, self.r + self.Eb[idx].dot(self.Et))
        # E_q [log p(theta) - log q(theta)]
        bound += gamma_term(self.Et, self.Elogt, self.nu_t, self.rho_t,
                            self.a, self.b)
        # E_q [log p(beta) - log q(beta)]
        bound += gamma_term(self.Eb[idx], self.Elogb[idx], self.nu_b[idx],
                            self.rho_b[idx], self.c, self.d)
        return bound


class OnlineNegBinomMF(NegBinomMF):
    ''' Negtive-binomial matrix factorization with stochastic inference '''
    def __init__(self, n_components=100, batch_size=10, n_pass=10,
                 max_inner_iter=20, shuffle=True, tol=0.0001, init_flat_lam=True,
                 smoothness=100, random_state=None, verbose=False,
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
        max_inner_iter : int
            Maximal number of iterations to update local factors
        shuffle : bool
            Whether to shuffle the data or not
        init_temp : float
            The initial tempurature for determinstic annealing
        anneal_len : float
            The annealing length, measured in effective traversals of the
            dataset
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
        self.shuffle = shuffle
        self.tol = tol
        self.init_flat_lam = init_flat_lam
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

        self.r = float(kwargs.get('r', 1.0))

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
                if sparse.isspmatrix(mini_batch):
                    mini_batch = mini_batch.toarray()
                self.set_learning_rate(mini_batch, iter=count * n_items / self.batch_size + i)
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

        ratio = X / self._xexplog(idx=idx)
        self.nu_t = (1 - self.rho) * self.nu_t + self.rho * \
            (self.a + self._scale * np.exp(self.Elogt) * np.exp(self.Elogb[idx].T).dot(ratio))
        self.rho_t = (1 - self.rho) * self.rho_t + self.rho * \
            (self.b + self._scale * self.Eb[idx].T.dot(self.Elam))
        old_Et = self.Et.copy()
        self.Et, self.Elogt = comp_gamma_expectations(self.nu_t, self.rho_t)
        print '\tEt relative change: %.3f' % np.sqrt(np.sum((self.Et - old_Et)**2))
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
        bound = np.sum(X * np.log(self._xexplog(idx=idx)))
        bound += gamma_term(self.Elam, self.Eloglam, self.nu_lam, self.rho_lam,
                            self.r + X, self.r + self.Eb[idx].dot(self.Et))
        # E_q [log p(beta) - log q(beta)]
        bound += gamma_term(self.Eb[idx], self.Elogb[idx], self.nu_b[idx],
                            self.rho_b[idx], self.c, self.d)
        # scale the objective to "pretent" we have the full dataset
        bound *= self._scale
        # E_q [log p(theta) - log q(theta)]
        bound += gamma_term(self.Et, self.Elogt, self.nu_t, self.rho_t,
                            self.a, self.b)
        return bound


def comp_gamma_expectations(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def gamma_term(Ex, Elogx, nu, rho, a, b):
    ''' Compute E_q[log p(x) - log q(x)] where
    p(x) = Gamma(a, b) and q(x) = Gamma(nu, rho)
    '''
    score = ((a - nu) * Elogx + (rho - b) * Ex + special.gammaln(nu)
             - nu * np.log(rho)).sum()
    return score
