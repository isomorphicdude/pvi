import jax.numpy as np
from jax.scipy.stats import norm
import jax.debug


class LReg:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def log_prior(self, x):
        """
        Standard Gaussian prior for logistic regression weights.
        """
        assert x.ndim == 1
        scale = (1 / self.alpha) ** 0.5
        log_prior =  norm.logpdf(
            x,
            loc=0.,
            scale=scale).sum(-1)
        return log_prior

    def log_likeli(self, x, y):
        """
        logit link function for logistic regression.
        log-likelihood= target * logit - log(1 + exp(logit))
        where logit = covariates @ x
        """
        target, covariates = y
        logit = covariates @ x
        assert target.ndim == 1, f"Shape of target is {target.shape}"
        assert logit.ndim == 0, f"Shape of logit is {logit.shape}"
        return target.squeeze(-1) * logit - np.logaddexp(0, logit)