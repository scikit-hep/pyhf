from .. import get_backend
from .utils import loglambdav


def qmu(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`q_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, as defiend in
    Equation (14) in :xref:`arXiv:1007.1727`.

    .. math::
       :nowrap:

       \begin{equation}
          q_{\mu} = \left\{\begin{array}{ll}
          -2\ln\lambda\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
          \end{array}\right.
        \end{equation}


    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (Tensor): The initial parameters
        par_bounds(Tensor): The bounds on the paramter values

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    tensorlib, optimizer = get_backend()
    mubhathat = optimizer.constrained_bestfit(
        loglambdav, mu, data, pdf, init_pars, par_bounds
    )
    muhatbhat = optimizer.unconstrained_bestfit(
        loglambdav, data, pdf, init_pars, par_bounds
    )
    qmu = loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf)
    qmu = tensorlib.where(
        muhatbhat[pdf.config.poi_index] > mu, tensorlib.astensor([0]), qmu
    )
    return qmu
