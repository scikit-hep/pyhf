import shlex

import pytest


@pytest.mark.usefixtures("tmp_path")
def test_2bin_1channel(script_runner):
    command = f"pyhf inspect {'docs/examples/json/2-bin_1-channel.json':s}"
    ret = script_runner.run(shlex.split(command))
    assert ret.success
