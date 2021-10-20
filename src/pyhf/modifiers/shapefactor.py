import logging

import pyhf
from pyhf import events
from pyhf.tensor.manager import get_backend
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': 'unconstrained',
        'n_parameters': len(sample_data),
        'is_shared': True,
        'is_scalar': False,
        'inits': (1.0,) * len(sample_data),
        'bounds': ((0.0, 10.0),) * len(sample_data),
        'fixed': False,
    }


class shapefactor_builder:
    """Builder class for collecting shapefactor modifier data"""

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        maskval = True if thismod else False
        mask = [maskval] * len(nom)
        return {'mask': mask}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]['data']['mask'] += moddata['mask']
        if thismod:
            self.required_parsets.setdefault(
                thismod['name'],
                [required_parset(defined_samp['data'], thismod['data'])],
            )

    def finalize(self):
        return self.builder_data


class shapefactor_combined:
    name = 'shapefactor'
    op_code = 'multiplication'

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        """

        Args:
            modifiers (:obj:`list` of :obj:`tuple`): List of tuples of
             form ``(modifier, modifier_type)``.
            pdfconfig (:class:`~pyhf.pdf._ModelConfig`): Configuration for the model.
            builder_data (:obj:`dict`): Map of keys ``'modifier_type/modifier'``
             to the channels and bins they are applied to.
            batch_size (:obj:`int`): The number of rows in the resulting tensor.
             If :obj:`None` defaults to ``1``.

        Imagine a situation where we have 2 channels (``SR``, ``CR``), 3 samples
        (``sig1``, ``bkg1``, ``bkg2``), and 2 :class:`~pyhf.modifiers.shapefactor`
        modifiers (``coupled_shapefactor``, ``uncoupled_shapefactor``).
        Let's say this is the set-up:

        .. code-block::

            SR(nbins=2)
              sig1 -> subscribes to normfactor
              bkg1 -> subscribes to coupled_shapefactor
            CR(nbins=3)
              bkg2 -> subscribes to coupled_shapefactor, uncoupled_shapefactor

        The ``coupled_shapefactor`` needs to have 3 nuisance parameters to account
        for the ``CR``, with 2 of them shared in the ``SR``.
        The ``uncoupled_shapefactor`` just has 3 nuisance parameters.

        ``self._parindices`` will look like

        .. code-block::

            [0, 1, 2, 3, 4, 5, 6]

        ``self._shapefactor_indices`` will look like

        .. code-block::

            [0, 1, 2, 3, 4, 5, 6]
            [[1,2,3],[4,5,6]]
             ^^^^^^^         = coupled_shapefactor
                     ^^^^^^^ = uncoupled_shapefactor

        with the 0th par-index corresponding to the
        :class:`~pyhf.modifiers.normfactor`. Because the ``SR`` channel has 2
        bins, and the ``CR`` channel has 3 bins (with ``SR`` before ``CR``),
        ``global_concatenated_bin_indices`` looks like

        .. code-block::

            [0, 1, 0, 1, 2]
            ^^^^^            = SR channel

                  ^^^^^^^^^  = CR channel

        So now we need to gather the corresponding
        :class:`~pyhf.modifiers.shapefactor` indices according to
        ``global_concatenated_bin_indices``. Therefore ``self._shapefactor_indices``
        now looks like

        .. code-block::

            [[1, 2, 1, 2, 3], [4, 5, 4, 5, 6]]

        and at that point can be used to compute the effect of
        :class:`~pyhf.modifiers.shapefactor`.
        """
        default_backend = pyhf.default_backend

        self.batch_size = batch_size
        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        shapefactor_mods = [m for m, _ in modifiers]

        parfield_shape = (self.batch_size or 1, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, shapefactor_mods
        )

        self._shapefactor_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]

        global_concatenated_bin_indices = [
            [[j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])]]
        ]

        self._access_field = default_backend.tile(
            global_concatenated_bin_indices,
            (len(shapefactor_mods), self.batch_size or 1, 1),
        )
        # access field is now
        # e.g. for a 3 channel (3 bins, 2 bins, 5 bins) model
        # [
        #   [0 1 2 0 1 0 1 2 3 4] (number of rows according to batch_size but at least 1)
        #   [0 1 2 0 1 0 1 2 3 4]
        #   [0 1 2 0 1 0 1 2 3 4]
        # ]

        # the index selection of param_viewer is a
        # list of (batch_size, par_slice) tensors
        # so self.param_viewer.index_selection[s][t]
        # points to the indices for a given systematic
        # at a given position in the batch
        # we thus populate the access field with these indices
        # up to the point where we run out of bins (in case)
        # the paramset slice is larger than the number of bins
        # in which case we use a dummy index that will be masked
        # anyways in apply (here: 0)

        # access field is shape (sys, batch, globalbin)
        for s, syst_access in enumerate(self._access_field):
            for t, batch_access in enumerate(syst_access):
                selection = self.param_viewer.index_selection[s][t]
                for b, bin_access in enumerate(batch_access):
                    self._access_field[s, t, b] = (
                        selection[bin_access] if bin_access < len(selection) else 0
                    )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.shapefactor_mask = tensorlib.tile(
            tensorlib.astensor(self._shapefactor_mask, dtype="bool"),
            (1, 1, self.batch_size or 1, 1),
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

        self.shapefactor_default = tensorlib.ones(
            tensorlib.shape(self.shapefactor_mask)
        )
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapefactor_mask)[1])

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        if self.batch_size is None:
            flat_pars = pars
        else:
            flat_pars = tensorlib.reshape(pars, (-1,))
        shapefactors = tensorlib.gather(flat_pars, self.access_field)
        results_shapefactor = tensorlib.einsum(
            'mab,s->msab', shapefactors, self.sample_ones
        )
        results_shapefactor = tensorlib.where(
            self.shapefactor_mask, results_shapefactor, self.shapefactor_default
        )
        return results_shapefactor
