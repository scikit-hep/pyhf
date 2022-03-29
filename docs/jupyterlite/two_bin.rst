.. tabs::

   .. tab:: Python

      .. code:: python

          import pyhf
          import numpy as np
          import matplotlib.pyplot as plt
          from pyhf.contrib.viz import brazil

          pyhf.set_backend("numpy")
          model = pyhf.simplemodels.uncorrelated_background(
              signal=[30.0, 45.0], bkg=[100.0, 150.0], bkg_uncertainty=[15.0, 20.0]
          )
          data = [100.0, 145.0] + model.config.auxdata

          poi_vals = np.linspace(0, 5, 41)
          results = [
              pyhf.infer.hypotest(
                  test_poi, data, model, test_stat="qtilde", return_expected_set=True
              )
              for test_poi in poi_vals
          ]

          fig, ax = plt.subplots()
          fig.set_size_inches(7, 5)
          brazil.plot_results(poi_vals, results, ax=ax)
          fig.show()

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
          import numpy as np
          import matplotlib.pyplot as plt
          from pyhf.contrib.viz import brazil

          pyhf.set_backend("numpy")
          model = pyhf.simplemodels.uncorrelated_background(
              signal=[30.0, 45.0], bkg=[100.0, 150.0], bkg_uncertainty=[15.0, 20.0]
          )
          data = [100.0, 145.0] + model.config.auxdata

          poi_vals = np.linspace(0, 5, 41)
          results = [
              pyhf.infer.hypotest(
                  test_poi, data, model, test_stat="qtilde", return_expected_set=True
              )
              for test_poi in poi_vals
          ]

          fig, ax = plt.subplots()
          fig.set_size_inches(7, 5)
          brazil.plot_results(poi_vals, results, ax=ax)
          fig
