from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as np

MIN_PROB_DPO = 1e-6
MAX_PROB_DPO = 1 - 1e-6


def irls(Phi, A, theta0, Lambda0, irls_error=1e-6, irls_num_iter=1000):
  # Iteratively reweighted least squares (IRLS) for the Plackett-Luce model
  # Implemented based on page 27 of https://arxiv.org/pdf/2301.11270.pdf
  n, num_actions, d = Phi.shape
  _, K = A.shape
  theta = np.copy(theta0)

  num_iter = 0
  while num_iter < irls_num_iter:
    theta_old = np.copy(theta)

    PhiA = np.zeros((n, K, d))
    for j in range(K):
      PhiA[:, j, :] = Phi[np.arange(n), A[:, j], :]

    negloglik = 0.0
    grad = np.zeros(d)
    Hessian = np.zeros((d, d))
    scores = np.einsum("tjk,k->tj", PhiA, theta)
    for j in range(K):
      scores[:, j :] -= scores[:, j :].max(axis=-1)[:, np.newaxis]
      norm = np.exp(scores[:, j :]).sum(axis=-1)
      norm2 = np.square(norm)

      negloglik -= (scores[:, j] - np.log(norm)).mean()
      for k in range(j, K):
        grad -= (np.exp(scores[:, k]) / norm).dot(PhiA[:, j, :] - PhiA[:, k, :]) / n
        for kp in range(j, K):
          Delta = PhiA[:, k, :] - PhiA[:, kp, :]
          Hessian += np.einsum("t,ti,tj->ij",
            0.5 * (np.exp(scores[:, k] + scores[:, kp]) / norm2), Delta, Delta) / n

    grad += Lambda0.dot(theta - theta0)
    Hessian += Lambda0

    theta = theta - np.linalg.inv(Hessian).dot(grad)

    if np.linalg.norm(theta - theta_old) < irls_error:
      break;
    num_iter += 1

  converged = (num_iter < irls_num_iter)
  return theta, grad, Hessian, converged


class PlackettLuce:
  def __init__(self, Phi):
    self.Phi = np.copy(Phi)
    self.n, self.L, self.d = Phi.shape

  def loglik(self, A, theta, pos=0):
    '''
    Inputs
      A: n x K matrix of K answers in n interactions
      theta: PL model parameter
      pos: number of positions for the loglik computation
    Output
      logliks: loglik of answers for each interaction
    '''
    K = A.shape[-1]
    if pos == 0:
      pos = K

    PhiA = np.zeros((self.n, K, self.d))
    for i in range(K):
      PhiA[:, i, :] = self.Phi[np.arange(self.n), A[:, i], :]
    PhiA = jnp.asarray(PhiA)

    logliks = jnp.zeros(self.n)
    scores = jnp.einsum("tij,j->ti", PhiA, theta)
    for i in range(pos):
      max_score = stop_gradient(scores[:, i :].max(axis=-1))
      norm = jnp.exp(scores[:, i :] - max_score[:, jnp.newaxis]).sum(axis=-1)
      logliks += scores[:, i] - max_score - jnp.log(norm)
    return logliks

  def sample(self, K, theta, num_samples=1):
    '''
    Inputs
      K: number of sampled answers per interaction
      theta: PL model parameter
    Output
      A: n x num_samples x K matrix of sampled answers
    '''
    A = np.zeros((self.n, num_samples, K), dtype=int)

    scores = np.einsum("tij,j->ti", self.Phi, theta)
    scores -= scores.max(axis=-1)[:, np.newaxis]
    p = np.exp(scores) + 1e-64
    p /= p.sum(axis=-1)[:, np.newaxis]
    for t in range(self.n):
      for sample in range(num_samples):
        A[t, sample, :] = np.random.choice(self.L, K, replace=False, p=p[t, :])

    if num_samples == 1:
      A = np.squeeze(A, axis=1)

    return A

  def mle(self, A, theta0=None):
    '''
    Inputs
      A: n x K matrix of K answers in n interactions
      theta0: initial PL model parameter
    Output
      theta: learned PL model parameter
    '''
    if theta0 is None:
      theta0 = np.zeros(self.d)

    theta_hat, _, _, converged = irls(self.Phi, A, theta0, 1e-3 * np.eye(self.d))
    if not converged:
      raise Exception("Plackett-Luce learning failed.")

    return theta_hat