from .. import get_backend


def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    _, optimizer = get_backend()
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds
    )
    return pdf.expected_data(bestfit_nuisance_asimov)
