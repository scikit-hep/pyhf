.. tabs::

   .. tab:: Python

      .. code:: pycon

         >>> import pyhf
         >>> pyhf.set_backend("numpy")
         >>> wspace = pyhf.Workspace(requests.get("https://git.io/JJYDE").json())
         >>> model = wspace.model()
         >>> data = wspace.data(model)
         >>> test_mu = 1.0
         >>> CLs_obs, CLs_exp = pyhf.infer.hypotest(
         ...     test_mu, data, model, test_stat="qtilde", return_expected=True
         ... )
         >>> print(f"Observed: {CLs_obs:.8f}, Expected: {CLs_exp:.8f}")
         Observed: 0.35998409, Expected: 0.35998409

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
          import pyodide
          import json
          import pyhf

          pyhf.set_backend("numpy")
          spec = pyodide.open_url("https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json")
          wspace = pyhf.Workspace(json.load(spec))
          model = wspace.model()
          data = wspace.data(model)
          test_mu = 1.0
          CLs_obs, CLs_exp = pyhf.infer.hypotest(
             test_mu, data, model, test_stat="qtilde", return_expected=True
          )
          print(f"Observed: {CLs_obs:.8f}, Expected: {CLs_exp:.8f}")
