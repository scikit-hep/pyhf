{
    simple: {
        local model = self.source,
        local channel = "singlechannel",
        "channels": [
            {
                "name": channel,
                "samples": [
                    {
                        "data": model.bindata.sig,
                        "modifiers": [
                            {
                                "data": null,
                                "name": "mu",
                                "type": "normfactor"
                            }
                        ],
                        "name": "signal"
                    },
                    {
                        "data": model.bindata.bkg,
                        "modifiers": [
                            {
                                "data": {
                                    "lo": 0.9,
                                    "hi": 1.1
                                },
                                "name": "bkg_norm",
                                "type": "normsys"
                            }
                        ],
                        "name": "background"
                    }
                ]
            }
        ],
        "data" : {
            [channel]: model.bindata.data
        },
        "toplvl": {
            "measurements": [
            {"config": {"poi": "mu"}, "name": "HelloWorld"}
            ]
        },
    },
    histosys_test: {
        local model = self.source,
        local channel = "singlechannel",
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': model.bindata.sig,
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': null
                            }
                        ]
                    },
                    {
                        'name': 'background',
                        'data': model.bindata.bkg,
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': model.bindata.bkgsys_dn,
                                    'hi_data': model.bindata.bkgsys_up
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    },
    two_bin_two_channel_coupledhistosys_test: {
        local model = self.source,
        local channel = "singlechannel",
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': model.channels.signal.bindata.sig,
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': null
                            }
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': model.channels.signal.bindata.bkg1,
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': model.channels.signal.bindata.bkg1_dn,
                                    'hi_data': model.channels.signal.bindata.bkg1_up
                                }
                            }
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': model.channels.signal.bindata.bkg2,
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': model.channels.signal.bindata.bkg2_dn,
                                    'hi_data': model.channels.signal.bindata.bkg2_up
                                }
                            }
                        ]
                    }
                ]
            },
            {
                'name': 'control',
                'samples': [
                    {
                        'name': 'background',
                        'data': model.channels.control.bindata.bkg1,
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': model.channels.control.bindata.bkg1_dn,
                                    'hi_data': model.channels.control.bindata.bkg1_up
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
}