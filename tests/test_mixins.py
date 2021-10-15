import pyhf
import pyhf.readxml
import pytest


@pytest.fixture(
    scope='session',
    params=[
        ('validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/')
    ],
    ids=['example-one'],
)
def spec(request):
    return pyhf.readxml.parse(*request.param)


def test_channel_summary_mixin(spec):
    assert 'channels' in spec
    mixin = pyhf.mixins._ChannelSummaryMixin(channels=spec['channels'])
    assert mixin.channel_nbins == {'channel1': 2}
    assert mixin.channels == ['channel1']
    assert mixin.modifiers == [
        ('SigXsecOverSM', 'normfactor'),
        ('lumi', 'lumi'),
        ('staterror_channel1', 'staterror'),
        ('syst1', 'normsys'),
        ('syst2', 'normsys'),
        ('syst3', 'normsys'),
    ]
    assert mixin.samples == ['background1', 'background2', 'signal']


def test_channel_summary_mixin_empty():
    mixin = pyhf.mixins._ChannelSummaryMixin(channels=[])
    assert mixin.channel_nbins == {}
    assert mixin.channels == []
    assert mixin.modifiers == []
    assert mixin.samples == []


def test_channel_nbins_sorted_as_channels(spec):
    assert "channels" in spec
    spec["channels"].append(spec["channels"][0].copy())
    spec["channels"][-1]["name"] = "a_make_first_in_sort_channel2"
    mixin = pyhf.mixins._ChannelSummaryMixin(channels=spec["channels"])
    assert mixin.channels == ["a_make_first_in_sort_channel2", "channel1"]
    assert list(mixin.channel_nbins.keys()) == mixin.channels
