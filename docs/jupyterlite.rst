Try out now with JupyterLite_
-----------------------------

.. admonition:: Click below to try pyhf in the browser:
    :class: dropdown

     #. Type (or copy and paste) code in the input cell.
     #. To execute the code, press ``Shift + Enter`` or click on the (Run) ▶ button in the toolbar.

     To get going try copying and pasting the "Hello World" example below!

..
  Comment: Use https://github.com/jupyterlite/jupyterlite-sphinx

.. replite::
   :kernel: python
   :height: 600px
   :prompt: Try pyhf!
   :prompt_color: #dc3545

   import piplite
   # Install pyhf in the browser
   await piplite.install(["pyhf==0.7.1", "matplotlib>=3.0.0"])
   %matplotlib inline
   import pyhf
   # You can now use pyhf!

..
  Comment: Add an extra blank line as a spacer

|

.. _JupyterLite: https://jupyterlite.readthedocs.io/
