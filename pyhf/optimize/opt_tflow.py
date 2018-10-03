import logging
import numpy as np
import tensorflow as tf
from tensorflow.errors import InvalidArgumentError

log = logging.getLogger(__name__)

class tflow_optimizer(object):
    def __init__(self, tensorlib):
        self.tb = tensorlib
        self.relax = 0.1
        self.maxit = 1000
        self.eps = 1e-4

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        #the graph
        data      = self.tb.astensor(data)
        parlist   = [self.tb.astensor([p]) for p in init_pars]

        pars      = self.tb.concatenate(parlist)
        objective = objective(pars,data,pdf)
        hessian   = tf.hessians(objective, pars)[0]+1e-10
        gradient  = tf.gradients(objective, pars)[0]
        invhess   = tf.linalg.inv(hessian)
        update    = tf.transpose(tf.matmul(invhess, tf.transpose(tf.stack([gradient]))))[0]

        # print(self.tb.session.run(hessian, feed_dict={pars: init_pars}))

        #run newton's method
        best_fit = init_pars
        for i in range(self.maxit):
            try:
                up = self.tb.session.run(update, feed_dict={pars: best_fit})
                best_fit = best_fit-self.relax*up
                if np.abs(np.max(up)) < self.eps:
                    break
            except InvalidArgumentError:
                o,p,g,h = self.tb.session.run([
                    objective,
                    pars,
                    gradient,
                    hessian,
                ], feed_dict={best_fit: best_fit})
                raise

        return best_fit.tolist()

    def constrained_bestfit(self, objective, constrained_mu, data, pdf, init_pars, par_bounds):
        #the graph
        data      = self.tb.astensor(data)

        nuis_pars = [self.tb.astensor([p]) for i,p in enumerate(init_pars) if i!=pdf.config.poi_index]
        poi_par   = self.tb.astensor([constrained_mu])

        nuis_cat = self.tb.concatenate(nuis_pars)
        pars = self.tb.concatenate([nuis_cat[:0],poi_par,nuis_cat[0:]])
        objective = objective(pars,data,pdf)
        hessian   = tf.hessians(objective, nuis_cat)[0]+1e-10
        gradient  = tf.gradients(objective, nuis_cat)[0]
        invhess   = tf.linalg.inv(hessian)
        update    = tf.transpose(tf.matmul(invhess, tf.transpose(tf.stack([gradient]))))[0]

        #run newton's method
        best_fit_nuis = [x for i,x in enumerate(init_pars) if i!= pdf.config.poi_index]
        for i in range(self.maxit):
            try:
                up = self.tb.session.run(update, feed_dict={nuis_cat: best_fit_nuis})
                best_fit_nuis = best_fit_nuis-self.relax*up
                if np.abs(np.max(up)) < self.eps:
                    break
            except InvalidArgumentError:
                o,p,g,up,h = self.tb.session.run([
                    objective,
                    pars,
                    gradient,
                    hessian,
                ], feed_dict={nuis_cat: best_fit_nuis})
                log.error('----- Constrained Fit -----\nIteration: {}\nObjective: {}\nPars: {}\nGradient: {}\nHessias was: {}'.format(
                    i,
                    self.tb.tolist(o),
                    self.tb.tolist(p),
                    self.tb.tolist(g),
                    self.tb.tolist(h),
                ))
                raise

        best_fit = best_fit_nuis.tolist()
        best_fit.insert(pdf.config.poi_index,constrained_mu)
        return best_fit
