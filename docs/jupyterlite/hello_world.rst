.. tabs::

   .. tab:: Python

      .. code:: pycon

         >>> import pyhf
         >>> pyhf.set_backend("numpy")
         >>> model = pyhf.simplemodels.uncorrelated_background(
         ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
         ... )
         >>> data = [51, 48] + model.config.auxdata
         >>> test_mu = 1.0
         >>> CLs_obs, CLs_exp = pyhf.infer.hypotest(
         ...     test_mu, data, model, test_stat="qtilde", return_expected=True
         ... )
         >>> print(f"Observed: {CLs_obs:.8f}, Expected: {CLs_exp:.8f}")
         Observed: 0.05251497, Expected: 0.06445321

   .. tab:: JupyterLite

      .. replite::
         :kernel: python
         :toolbar: 1
         :theme: JupyterLab Light
         :width: 100%
         :height: 600px

         import piplite
         await piplite.install(["pyhf==0.6.3", "requests"])

         %matplotlib inline
         import pyhf

         pyhf.set_backend("numpy")
         model = pyhf.simplemodels.uncorrelated_background(
            signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
         )
         data = [51, 48] + model.config.auxdata
         test_mu = 1.0
         CLs_obs, CLs_exp = pyhf.infer.hypotest(
            test_mu, data, model, test_stat="qtilde", return_expected=True
         )
         print(f"Observed: {CLs_obs:.8f}, Expected: {CLs_exp:.8f}")
