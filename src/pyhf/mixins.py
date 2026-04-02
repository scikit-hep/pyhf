from __future__ import annotations

import logging
from typing import Any
from collections.abc import Sequence

from pyhf import exceptions
from pyhf.typing import Channel

log = logging.getLogger(__name__)


class _ChannelSummaryMixin:
    """
    A mixin that provides summary data of the provided channels.

    This mixin will forward all other information to other classes defined in the Child class.

    Args:
      **channels: A list of channels to provide summary information about. Follows the `defs.json#/definitions/channel` schema.
    """

    def __init__(self, *args: Any, **kwargs: Sequence[Channel]):
        channels = kwargs.pop('channels')
        super().__init__(*args, **kwargs)

        # check for duplicates
        self._check_for_duplicates(channels)

        self._channels: list[str] = []
        self._samples: list[str] = []
        self._modifiers: list[tuple[str, str]] = []
        # keep track of the width of each channel (how many bins)
        self._channel_nbins: dict[str, int] = {}
        # need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        for channel in channels:
            self._channels.append(channel['name'])
            self._channel_nbins[channel['name']] = len(channel['samples'][0]['data'])
            for sample in channel['samples']:
                self._samples.append(sample['name'])
                for modifier_def in sample['modifiers']:
                    self._modifiers.append(
                        (
                            modifier_def['name'],  # mod name
                            modifier_def['type'],  # mod type
                        )
                    )

        self._channels = sorted(list(set(self._channels)))
        self._samples = sorted(list(set(self._samples)))
        self._modifiers = sorted(list(set(self._modifiers)))
        self._channel_nbins = {
            channel: self._channel_nbins[channel] for channel in self._channels
        }

        self._channel_slices = {}
        begin = 0
        for c in self._channels:
            end = begin + self._channel_nbins[c]
            self._channel_slices[c] = slice(begin, end)
            begin = end

    @property
    def channels(self) -> list[str]:
        """
        Ordered list of channel names in the model.
        """
        return self._channels

    @property
    def samples(self) -> list[str]:
        """
        Ordered list of sample names in the model.
        """
        return self._samples

    @property
    def modifiers(self) -> list[tuple[str, str]]:
        """
        Ordered list of pairs of modifier name/type in the model.
        """
        return self._modifiers

    @property
    def channel_nbins(self) -> dict[str, int]:
        """
        Dictionary mapping channel name to number of bins in the channel.
        """
        return self._channel_nbins

    @property
    def channel_slices(self) -> dict[str, slice]:
        """
        Dictionary mapping channel name to the bin slices in the model.
        """
        return self._channel_slices

    def _check_for_duplicates(self, channels: Sequence[Channel]) -> None:
        """
        Check for duplicate channels.
        Check for duplicate samples within each channel.
        Check for duplicate modifiers within each sample.
        """
        channel_names = [channel['name'] for channel in channels]
        if len(channel_names) != len(set(channel_names)):
            duplicates = sorted(
                set([f"'{x}'" for x in channel_names if channel_names.count(x) > 1])
            )
            raise exceptions.InvalidModel(
                "Duplicate channels "
                + ", ".join(duplicates)
                + " found in the model. Rename one of them."
            )
        for channel in channels:
            sample_names = [samples['name'] for samples in channel['samples']]
            if len(sample_names) != len(set(sample_names)):
                duplicates = sorted(
                    set([f"'{x}'" for x in sample_names if sample_names.count(x) > 1])
                )
                raise exceptions.InvalidModel(
                    "Duplicate samples "
                    + ", ".join(duplicates)
                    + f" found in the channel '{channel['name']}'. Rename one of them."
                )
            for sample in channel['samples']:
                modifiers = [
                    (modifier['name'], modifier['type'])
                    for modifier in sample['modifiers']
                ]
                if len(modifiers) != len(set(modifiers)):
                    duplicates = sorted(
                        set([f"'{x[0]}'" for x in modifiers if modifiers.count(x) > 1])
                    )
                    raise exceptions.InvalidModel(
                        "Duplicate modifiers "
                        + ", ".join(duplicates)
                        + f" of the same type found in channel '{channel['name']}' and sample '{sample['name']}'. Rename one of them."
                    )
