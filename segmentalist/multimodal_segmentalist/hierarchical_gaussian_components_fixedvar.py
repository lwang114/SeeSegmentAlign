"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import logging
import math
import numpy as np
from scipy.special import logsumexp
import scipy.signal as signal
logger = logging.getLogger(__name__)

import _cython_utils

EPS = 1e-100
#-----------------------------------------------------------------------------#
#                   FIXED VARIANCE GAUSSIAN COMPONENTS CLASS                  #
#-----------------------------------------------------------------------------#
class HierarchicalGaussianComponentsFixedVar(object):
    """
    Components of a Bayesian Gaussian mixture model (GMM) with fixed diagonal
    covariance matrices.

    This class is used to present the `K` components of a Bayesian GMM with
    fixed diagonal covariance matrices. All values necessary for computing
    likelihood terms are stored. For example, `mu_N_numerators` is a KxD matrix
    in which each D-dimensional row vector is the numerator for the posterior
    mu_N term in (30) in Murphy's bayesGauss notes. A NxD data matrix `X` and a
    Nx1 assignment vector `assignments` are also attributes of this class. In
    the member functions, `i` generally refers to the index of a data vector
    while `k` refers to the index of a mixture component.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    prior : `FixedVarPrior`
        Contains the fixed variance Dx1 vector `var`, the prior mean Dx1 vector
        `mu_0` and the prior variance Dx1 vector `var_0`.
    assignments : Nx1 vector of int
        The initial component assignments. If this values is None, then all
        data vectors are left unassigned indicated with -1 in the vector.
        Components should be labelled from 0.
    lm : e.g. BigramSmoothLM
        If specified, this language model is tied to the components and if
        a component is deleted, the counts of the language model is updated.
    M_max : int
        The maximum number of subword components 
    embed_technique : str
        Method by which the data vectors are created. Has to be one of `resample`, `interpolate`, `mean`, `rasanen`
    temporal_dim : int
        Temporal length of each embedding

    Global attributes
    -----------------
    N : int
        Number of data vectors.
    D : int 
        Dimensionality of data vectors.
    K : int
        Number of Gaussian word components.
    K : int
        Number of Gaussian subword components.
    T : int
        Number of time steps for each data vector.


    Component attributes
    --------------------
    mu_N_numerators : KxD matrix
        The numerator of (30) in Murphy's bayesGauss notes, p.3 for each
        component.
    precision_Ns : KxD matrix
        The precisions of the posterior distributions for each component given
        in (29) in Murphy's bayesGauss notes, p.3.
    log_prod_precision_preds : Kx1 vector
        The log of the product of the D precisions of the posterior predictive
        distribution in (40) in Murphy's bayesGauss, p.4 notes for each of the
        K components.
    precision_preds : KxD matrix
        Each D-dimensional row vector is the precisions for one of the K
        components.
    counts : Kx1 vector of int
        Counts for each of the K word components.
    phone_counts : Mx1 vector of int
        Counts for each of the M subword components.

    """

    def __init__(self, X, prior, lengths, assignments=None, K_max=None, lm=None, M_max=None, embed_technique='resample', T=10):

        # Attributes from parameters
        self.X = X
        self.lengths = lengths
        self.precision = 1./prior.var
        self.mu_0 = prior.mu_0
        self.precision_0 = 1./prior.var_0
        self.N, self.D = X.shape
        self.T = T
        if K_max is None:
            # assert False, "To-do: remove this, always require `K_max`"
            K_max = 1
        self.K_max = K_max

        if M_max is None:
          M_max = 1
        self.M_max = M_max
        self.lm = lm
        self.technique = embed_technique

        # Initialize attributes
        self.mu_N_numerators = np.zeros((self.M_max, self.D), np.float)
        self.precision_Ns = np.zeros((self.M_max, self.D), np.float)
        self.log_prod_precision_preds = np.zeros(self.M_max, np.float)
        self.precision_preds = np.zeros((self.M_max, self.D), np.float)
        self.counts = np.zeros(self.K_max, np.int)
        self.phone_counts = np.zeros(self.M_max, np.int)

        # Perform caching
        self._cache()

        # Initialize components based on `assignments`
        self.K = 0
        self.M = 0
        self.word_to_idx = {} # Mapping word tokens to their phone cluster indices   
        self.phone_to_idx = {}
        self.idx_to_word = []
        self.idx_to_phone = []
        if assignments is None:
            self.assignments = -1*np.ones(self.N, np.int)
        else:
            # Check that assignments are valid
            assignments = np.asarray(assignments, np.int)
            assert (self.N, ) == assignments.shape
            # Apart from unassigned (-1), components should be labelled from 0
            assert set(assignments).difference([-1]) == set(range(assignments.max() + 1))
            self.assignments = assignments

            # Add the data items
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_item(i, k)

    def _cache(self):
        self._cached_neg_half_D_log_2pi = -0.5*self.D*math.log(2.*np.pi)
        self.cached_log_prior = np.zeros(self.N, np.float)
        for i in xrange(self.N):
            self.cached_log_prior[i] = self.log_prior(i)

    def cache_component_stats(self, k):
        """
        Return the statistics for component `k` in a tuple.

        In this way the statistics for a component can be cached and can then
        be restored later using `restore_component_from_stats`.
        """
        return (
            self.mu_N_numerators[k].copy(),
            self.precision_Ns[k].copy(),
            self.log_prod_precision_preds[k],
            self.precision_preds[k].copy(),
            self.counts[k]
            )

    def restore_component_from_stats(
            self, k, mu_N_numerator, precision_N, log_prod_precision_pred, precision_pred, count
            ):
        """Restore component `k` using the provided statistics."""
        self.mu_N_numerators[k, :] = mu_N_numerator
        self.precision_Ns[k, :] = precision_N
        self.log_prod_precision_preds[k] = log_prod_precision_pred
        self.precision_preds[k, :] = precision_pred
        self.counts[k] = count

    def add_item(self, i, k):
        """
        Add data vector `X[i]` to word component `k`. If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """
        assert not i == -1
        if isinstance(k, int):
          k = [k]
        assert self.lengths[i] == len(k)
        
        for m in k: 
          if m == self.M: # Create a new phone component 
            if self.M == self.M_max: # Check the number of subword units is below upper limit; otherwise, increase the upper limit
              self.M_max += 1
              self.phone_counts = np.append(self.phone_counts, np.zeros(1))
              self.mu_N_numerators = np.append(self.mu_N_numerators, np.zeros((1, self.D)))
              self.precision_preds = np.append(self.precision_preds, np.zeros((1, self.D)))

            for im in range(self.M_max): # Create a key for the new component
              if not str(im) in self.phone_to_idx:  
                # print('In HG add_item, phone_counts[M] before being replaced: ' + str(self.phone_counts[self.M]))
                # print('In HG add_item, add a new phone: ' + str(im) + ' ' + str(self.M))
                self.phone_to_idx[str(im)] = self.M
                self.idx_to_phone.append(str(im)) 
                self.M += 1
                assert self.M == len(self.idx_to_phone)
                break 
          
            self.mu_N_numerators[m, :] = self.precision_0*self.mu_0
            self.precision_Ns[m, :] = self.precision_0
        
        w = ','.join(self.idx_to_phone[m] for m in k)
        if not w in self.word_to_idx: # Create a new word component         
          # print('In HG add_item, add a new word: ' + w + ' ' + str(self.K))
          if self.K == self.K_max: # Check the number of word units is below upper limit
            self.K_max += 1
            self.counts = np.append(self.counts, np.zeros(1, np.int)) 
          self.word_to_idx[w] = self.K
          self.idx_to_word.append(w)
          self.K += 1
          assert self.K == len(self.idx_to_word)
        
        Xr = self.inv_embed(i)
        for l, m in enumerate(k): # Update component stats
          self.mu_N_numerators[m, :] += self.precision*Xr[l]
          self.precision_Ns[m, :] += self.precision
          self.phone_counts[m] += 1
          self._update_log_prod_precision_pred_and_precision_pred(m)
        
        self.counts[self.word_to_idx[w]] += 1
        self.assignments[i] = self.word_to_idx[w]

    def del_item(self, i):
        """Remove data vector `X[i]` from its component."""
        assert not i == -1
        k = self.assignments[i]
        word = self.idx_to_word[k]
        L = self.lengths[i] 

        # Only do something if the data vector has been assigned
        if k != -1:
            w_indices = [self.phone_to_idx[m] for m in word.split(',')] 
            assert L == len(w_indices) 
            self.counts[k] -= 1
            for l, m_idx in enumerate(w_indices):
              self.phone_counts[m_idx] -= 1

            if self.counts[k] == 0:
              self.del_component(w_indices)
                      
            self.assignments[i] = -1
       
            Xr = self.inv_embed(i)
            for l, m in enumerate(word.split(',')):
              m_idx = self.phone_to_idx[m]
              if self.phone_counts[m_idx] == 0: # Delete the phone component if it is empty 
                self.del_phone_component(m_idx)          
              else: # Update the component stats if it is not empty 
                self.mu_N_numerators[m_idx, :] -= self.precision*Xr[l]
                self.precision_Ns[m_idx, :] -= self.precision
                self._update_log_prod_precision_pred_and_precision_pred(m_idx)

    def del_component(self, k):
        """Remove the word component `k`, where `k` is a phone index or list of phone indices.""" 
        self.K -= 1
        if isinstance(k, int): # If k is an integer, delete the phone component `k` 
          self.del_phone_component(k) 
        else: # Else delete the word component k and every empty phone component in k
          w = ','.join([self.idx_to_phone[m] for m in k])
          # print('In HG del_component, delete word: ' + w + ' ' + str(self.word_to_idx[w]))
          if self.word_to_idx[w] != self.K:
            self.counts[self.word_to_idx[w]] = self.counts[self.K]
            self.idx_to_word[self.word_to_idx[w]] = self.idx_to_word[self.K]
            self.word_to_idx[self.idx_to_word[self.K]] = self.word_to_idx[w]
            self.assignments[np.where(self.assignments == self.K)] = self.word_to_idx[w] 
           
          del self.idx_to_word[self.K]
          self.word_to_idx.pop(w) 
          self.counts[self.K] = 0
          # print('%s is in the dictionary? ' % w + str(w in self.word_to_idx or w in self.idx_to_word))

        
    def del_phone_component(self, m):
        """Remove the subword component `m`."""
        logger.debug("Deleting subword component " + str(m))
        # print('In HG del_phone, delete phone: ' + str(self.idx_to_phone[m]) + ' ' + str(m))
        self.M -= 1
        phone = self.idx_to_phone[m]
        if m != self.M:
            # Put stats from last component into place of the one being removed
            self.mu_N_numerators[m] = self.mu_N_numerators[self.M]
            self.precision_Ns[m, :] = self.precision_Ns[self.M, :]
            self.log_prod_precision_preds[m] = self.log_prod_precision_preds[self.M]
            self.precision_preds[m, :] = self.precision_preds[self.M, :]
            self.phone_counts[m] = self.phone_counts[self.M]
            self.phone_to_idx[self.idx_to_phone[self.M]] = m
            self.idx_to_phone[m] = self.idx_to_phone[self.M]

        self.phone_to_idx.pop(phone)     
        del self.idx_to_phone[self.M]   
        self.phone_counts[self.M] = 0
        assert len(self.idx_to_phone) == len(self.phone_to_idx) == self.M
     
        # Empty out stats for last component
        self.mu_N_numerators[self.M].fill(0.)
        self.precision_Ns[self.M, :].fill(0.)
        self.log_prod_precision_preds[self.M] = 0.
        self.precision_preds[self.M, :].fill(0.)
        self.phone_counts[self.M] = 0

        # print('%s is in the dictionary? ' % phone + str(phone in self.phone_to_idx or phone in self.idx_to_phone))

    # @profile
    def log_prior(self, i):
        """Return a lengths[i] vector of the probability of `X[i]` under the prior alone."""
        mu = self.mu_0 
        precision = self.precision_0
        log_prod_precision_pred = _cython_utils.sum_log(precision)
        precision_pred = precision
        return self._log_prod_norm(i, mu, log_prod_precision_pred, precision_pred)

    def log_post_pred_k(self, i, k):
        """
        Return the log posterior predictive probability of `X[i]` under
        component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k] 
        return self._log_prod_norm(i, mu_N, self.log_prod_precision_preds[k], self.precision_preds[k])

    # @profile
    def log_post_pred_active(self, i, log_post_preds):
        active_words = [w for w, iw in sorted(self.word_to_idx.items(), key=lambda x:x[1]) if self.lengths[i] == len(w.split(','))]
        return np.array([sum(log_post_preds[l, self.phone_to_idx[m]] for l, m in enumerate(w.split(','))) for w in active_words])

    # @profile
    def log_post_pred_inactive(self, log_post_preds, log_post_preds_active): 
        log_marg_i = logsumexp(log_post_preds, axis=1).sum() 
        if not len(log_post_preds_active): # If none of the components are active, return the whole log marginal
          return log_marg_i
        else:
          log_marg_i_active = logsumexp(log_post_preds_active)
          # print('log_marg_i, log_marg_i_active: ', log_marg_i, log_marg_i_active)
          # print('np.log(1 - np.exp(log_marg_i_active - log_marg_i)): ', np.log(max(1 - np.exp(log_marg_i_active - log_marg_i), EPS))) # XXX
          return log_marg_i + np.log(max(1 - np.exp(log_marg_i_active - log_marg_i), EPS)) # TODO Check this  

    # @profile
    def log_post_pred(self, i):
        """
        If vectorize is False, return a lengths[i]x`M`-dimensional matrix of the posterior predictive of `X[i]`
        under all components. 
        If vectorize is True, return a length `K`+1 vector with the first K entries containing log posterior predictive of `X[i]` under the active components and the last entry combining those of inactive components  
        """
        n_min = max(2, int(self.T / self.lengths[i]))
        n_max = n_min + 1
        # TODO Divide the intervals proportional to the durations of the segments
        if abs(n_max * self.lengths[i] - self.T) > abs(n_min * self.lengths[i] - self.T):
          n = n_min
        else:
          n = n_max     
        Dr = int(self.D / self.T * n)
        Xr = self.embed(self.X[i], n*self.lengths[i])  
        mu_Ns = np.asarray([self.embed(self.mu_N_numerators[m]/self.precision_Ns[m], n) for m in range(self.M)])
        precision_preds = np.asarray([self.embed(self.precision_preds[m], n) for m in range(self.M)])
        log_precision_preds = self.log_prod_precision_preds[:self.M]

        log_post_preds = np.nan*np.ones((self.lengths[i], self.M_max))
        for l in range(self.lengths[i]):
          deltas = mu_Ns - Xr[l*Dr:(l+1)*Dr]
          log_post_preds[l, :self.M] = (
              self._cached_neg_half_D_log_2pi
              + 0.5*log_precision_preds
              - 0.5*((deltas*deltas)*precision_preds).sum(axis=1)
              )

        # return (
        #     self._cached_neg_half_D_log_2pi
        #     + 0.5*self.log_prod_precision_preds[:self.K]
        #     - 0.5*(np.square(deltas)*self.precision_preds[:self.K]).sum(axis=1)
        #     ) 
        log_post_preds[:, self.M:] = np.tile(self.log_prior(i), (1, self.M_max-self.M))
        return log_post_preds

    # TODO
    def log_marg_k(self, k):
        """
        Return the log marginal probability of the data vectors assigned to
        component `k`.

        The log marginal probability p(X) = p(x_1, x_2, ..., x_N) is returned
        for the data vectors assigned to component `k`. See (55) in Murphy's
        bayesGauss notes, p. 15.
        """
        X = self.X[np.where(self.assignments == k)]
        N = self.counts[k]
        return np.sum(
            (N - 1)/2.*np.log(self.precision)
            - 0.5*N*math.log(2*np.pi)
            - 0.5*np.log(N/self.precision_0 + 1./self.precision)
            - 0.5*self.precision*np.square(X).sum(axis=0)
            - 0.5*self.precision_0*np.square(self.mu_0)
            + 0.5*(
                np.square(X.sum(axis=0))*self.precision/self.precision_0
                + np.square(self.mu_0)*self.precision_0/self.precision
                + 2*X.sum(axis=0)*self.mu_0
                )/(N/self.precision_0 + 1./self.precision)
            )  
    
    # TODO
    def log_marg(self):
        """
        Return the log marginal probability of all the data vectors given the
        component `assignments`.

        The log marginal probability of
        p(X|z) = p(x_1, x_2, ... x_N | z_1, z_2, ..., z_N) is returned.
        """
        log_prob_X_given_z = 0.
        for k in xrange(self.K):
            log_prob_X_given_z += self.log_marg_k(k)
        return log_prob_X_given_z

    def inv_embed(self, i):
        """
        Return a `n`x`D` matrix of phone-level embedding vectors reconstructed from data vector i
        """
        n = self.lengths[i] * self.T
        x = self.X[i].reshape(self.T, -1).T
        x_inv = self.__embed(x, n, technique=self.technique).T
        return x_inv.reshape(-1, self.D)

    def embed(self, x, n):
        return self.__embed(x.reshape(self.T, -1).T, n, technique=self.technique).T.flatten("C")

    def __embed(self, y, n, technique='resample'):
      if y.shape[1] < n and technique != 'mean': 
          technique = "resample"

      # Downsample
      if technique == "interpolate":
          x = np.arange(y.shape[1])
          f = interpolate.interp1d(x, y, kind="linear")
          x_new = np.linspace(0, y.shape[1] - 1, n)
          y_new = f(x_new)
      elif technique == "resample": 
          y_new = signal.resample(y.astype("float32"), n, axis=1)
      elif technique == "rasanen":
          # Taken from Rasenen et al., Interspeech, 2015
          d_frame = y.shape[0]
          n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
          y_new = np.mean(
              y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
              )
      elif technique == 'mean':
          if y.shape[1] == 0:
            return np.zeros((y.shape[0],)) 
          y_new = np.mean(y, axis=1, keepdims=True)
      return y_new


    def rand_k(self, k):
        """
        Return a random mean vector from the posterior product of normal
        distributions for component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k]
        var_N = 1./self.precision_Ns[k]
        mean = np.zeros(self.D)
        for i in range(self.D):
            mean[i] = np.random.normal(mu_N[i], np.sqrt(var_N[i]))
        return mean

    def get_assignments(self, list_of_i): # TODO Check this
        """
        Return a vector of the assignments for the data vector indices in
        `list_of_i`.
        """
        return [self.idx_to_word[i] for i in self.assignments[np.asarray(list_of_i)].tolist()]

    def _update_log_prod_precision_pred_and_precision_pred(self, k):
        """
        Update the precision terms for the posterior predictive distribution of
        component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k]
        precision_pred = self.precision_Ns[k]*self.precision / (self.precision_Ns[k] + self.precision) 
        self.log_prod_precision_preds[k] = np.log(precision_pred).sum()
        self.precision_preds[k, :] = precision_pred

    # @profile
    def _log_prod_norm(self, i, mu, log_prod_precision_pred, precision_pred): 
        """
        Return the value of the log of the product of univariate normal PDFs at
        `X[i]`.
        """ 
        n = max(2, int(self.T / self.lengths[i]))
        Dr = int(self.D / self.T * n)
        L = int(self.T / n) 

        log_prod_precision_preds_r = log_prod_precision_pred
        precision_pred_r = self.embed(precision_pred, n)
        log_probs = np.nan*np.ones((L, 1))
        for l in range(L):
          delta = self.X[i, l*Dr:(l+1)*Dr] - self.embed(mu, n)
          log_probs[l] = (
            self._cached_neg_half_D_log_2pi
            + 0.5 * log_prod_precision_preds_r
            - 0.5 * _cython_utils.sum_square_a_times_b(delta, precision_pred_r)
            )
        if L < self.lengths[i]: # For the probabilities of unused phone embeddings are equal to the probability of the last used phone embedding
          log_probs[L:] = log_probs[L-1] 
        return log_probs.sum() 

#-----------------------------------------------------------------------------#
#                     FIXED VARIANCE GAUSSIAN PRIOR CLASS                     #
#-----------------------------------------------------------------------------#

class FixedVarPrior(object):
    """
    The prior parameters for a fixed diagonal covariance multivariate Gaussian.
    """
    def __init__(self, var, mu_0, var_0):
        self.var = var
        self.mu_0 = mu_0
        self.var_0 = var_0


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def log_norm_pdf(x, mean, var):
    """Return the log of the normal PDF at `x`."""
    return -0.5*(np.log(2*np.pi) + np.log(var)) - 1./(2*var) * (x - mean)**2


def log_post_pred_unvectorized(gmm, i):
    """
    Return the same value as `GaussianComponentsFixedVar.log_post_pred` but
    using an unvectorized procedure, for testing purposes.
    """
    post_pred = np.zeros(gmm.K, np.float)
    for k in range(gmm.K):
        post_pred[k] = gmm.log_post_pred_k(i, k)
    return post_pred

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    pass


if __name__ == "__main__":
    main()
