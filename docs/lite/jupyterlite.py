# ---
# jupyter:
#   kernelspec:
#     display_name: Python (Pyodide)
#     language: python
#     name: python
# ---

# %% [markdown]
# # `pyhf` in the browser

# %% [markdown]
# * To run the code, click on the first cell (gray box) and press <kbd>Shift</kbd>+<kbd>Enter</kbd> or click on the (Run) ▶ button to run each cell.
# * Alternatively, from the `Run` menu select `Run All Cells`.
# * Feel free to experiment, and if you need to restore the original code reload this browser page. Any changes you make will be lost when you reload.
#
# To get going try copying and pasting the "Hello World" example below!

# %%
import piplite

# Install pyhf in the browser
await piplite.install(["pyhf==0.7.2", "matplotlib>=3.0.0"])
# %matplotlib inline
import pyhf

# You can now use pyhf!
