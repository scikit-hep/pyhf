import pyhf
import pyhf.cli
import pyhf.contrib.viz.brazil


def test_top_level_public_api():
    assert dir(pyhf) == [
        "Model",
        "PatchSet",
        "Workspace",
        "__version__",
        "exceptions",
        "get_backend",
        "infer",
        "interpolators",
        "modifiers",
        "optimize",
        "optimzier",
        "set_backend",
        "simplemodels",
        "tensor",
        "tensorlib",
        "utils",
    ]


def test_cli_public_api():
    assert dir(pyhf.cli) == ["cli", "complete", "contrib", "infer", "rootio", "spec"]


def test_constraints_public_api():
    assert dir(pyhf.constraints) == [
        "gaussian_constraint_combined",
        "poisson_constraint_combined",
    ]


def test_cli_contrib_public_api():
    assert dir(pyhf.cli.contrib) == ["download"]


def test_contrib_viz_public_api():
    assert dir(pyhf.contrib.viz.brazil) == [
        "BrazilBandCollection",
        "plot_brazil_band",
        "plot_cls_components",
        "plot_results",
    ]


def test_contrib_events_public_api():
    assert dir(pyhf.events) == [
        "Callables",
        "WeakList",
        "disable",
        "enable",
        "noop",
        "register",
        "subscribe",
        "trigger",
    ]


def test_contrib_exceptions_public_api():
    assert dir(pyhf.exceptions) == [
        "FailedMinimization",
        "ImportBackendError",
        "InvalidArchiveHost",
        "InvalidBackend",
        "InvalidInterpCode",
        "InvalidMeasurement",
        "InvalidModel",
        "InvalidModifier",
        "InvalidNameReuse",
        "InvalidOptimizer",
        "InvalidPatchLookup",
        "InvalidPatchSet",
        "InvalidPdfData",
        "InvalidPdfParameters",
        "InvalidSpecification",
        "InvalidTestStatistic",
        "InvalidWorkspaceOperation",
        "PatchSetVerificationError",
        "UnspecifiedPOI",
        "Unsupported",
    ]


def test_infer_public_api():
    assert dir(pyhf.infer) == [
        "calculators",
        "hypotest",
        "intervals",
        "mle",
        "test_statistics",
        "utils",
    ]


def test_infer_calculators_public_api():
    assert dir(pyhf.infer.calculators) == [
        "AsymptoticCalculator",
        "AsymptoticTestStatDistribution",
        "EmpiricalDistribution",
        "ToyCalculator",
        "generate_asimov_data",
    ]


def test_infer_intervals_public_api():
    assert dir(pyhf.infer.intervals) == ["upperlimit"]


def test_infer_mle_public_api():
    assert dir(pyhf.infer.mle) == ["fit", "fixed_poi_fit", "twice_nll"]


def test_infer_test_statistics_public_api():
    assert dir(pyhf.infer.test_statistics) == [
        "q0",
        "qmu",
        "qmu_tilde",
        "tmu",
        "tmu_tilde",
    ]


def test_infer_utils_public_api():
    assert dir(pyhf.infer.utils) == ["create_calculator", "get_test_stat"]


def test_interpolators_public_api():
    assert dir(pyhf.interpolators) == ["code0", "code1", "code2", "code4", "code4p"]


def test_modifiers_public_api():
    assert dir(pyhf.modifiers) == [
        "combined",
        "histosys",
        "histosys_combined",
        "lumi",
        "lumi_combined",
        "normfactor",
        "normfactor_combined",
        "normsys",
        "normsys_combined",
        "shapefactor",
        "shapefactor_combined",
        "shapesys",
        "shapesys_combined",
        "staterror",
        "staterror_combined",
    ]
