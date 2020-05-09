import logging

log = logging.getLogger(__name__)


class _ChannelSummaryMixin(object):
    """
    A mixin that provides summary data of the provided channels.

    This mixin will forward all other information to other classes defined in the Child class.

    Args:
      **channels: A list of channels to provide summary information about. Follows the `defs.json#/definitions/channel` schema.
    """

    def __init__(self, *args, **kwargs):
        channels = kwargs.pop('channels')
        super(_ChannelSummaryMixin, self).__init__(*args, **kwargs)
        self.channels = []
        self.samples = []
        self.parameters = []
        self.modifiers = []
        # keep track of the width of each channel (how many bins)
        self.channel_nbins = {}
        # need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        for channel in channels:
            self.channels.append(channel['name'])
            self.channel_nbins[channel['name']] = len(channel['samples'][0]['data'])
            for sample in channel['samples']:
                self.samples.append(sample['name'])
                for modifier_def in sample['modifiers']:
                    self.parameters.append(modifier_def['name'])
                    self.modifiers.append(
                        (
                            modifier_def['name'],  # mod name
                            modifier_def['type'],  # mod type
                        )
                    )

        self.channels = sorted(list(set(self.channels)))
        self.samples = sorted(list(set(self.samples)))
        self.parameters = sorted(list(set(self.parameters)))
        self.modifiers = sorted(list(set(self.modifiers)))
