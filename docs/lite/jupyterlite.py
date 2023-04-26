# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.1
# ---

# %% [markdown]
# * To run the code, click on the first cell (gray box) and press <kbd>Shift</kbd>+<kbd>Enter</kbd> or click on the (Run) ▶ button to run each cell.
# * Alternatively, from the `Run` menu select `Run All Cells`.
# * Feel free to experiment, but if you need to restore the original code reload this browser page. Any unsaved changes you make will be lost when you reload!

# %%
import piplite
# Install pyhf in the browser
await piplite.install(["pyhf==0.7.1", "matplotlib>=3.0.0"])
# %matplotlib inline
import pyhf
# You can now use pyhf!
