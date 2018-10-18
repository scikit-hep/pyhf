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
    }
}