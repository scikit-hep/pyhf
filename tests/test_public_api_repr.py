import pyhf
import pyhf.cli
import pyhf.contrib.utils
import pyhf.contrib.viz.brazil
import pyhf.readxml
import pyhf.writexml


def test_top_level_public_api():
    assert dir(pyhf) == [
        "Model",
        "PatchSet",
        "Workspace",
        "__version__",
        "compat",
        "default_backend",
        "exceptions",
        "get_backend",
        "infer",
        "interpolators",
        "modifiers",
        "optimizer",
        "parameters",
        "patchset",
        "pdf",
        "probability",
        "schema",
        "set_backend",
        "simplemodels",
        "tensor",
        "tensorlib",
        "utils",
        "workspace",
    ]


def test_cli_public_api():
    assert dir(pyhf.cli) == ["cli", "complete", "contrib", "infer", "rootio", "spec"]


def test_compat_public_api():
    assert dir(pyhf.compat) == ["interpret_rootname", "paramset_to_rootnames"]


def test_constraints_public_api():
    assert dir(pyhf.constraints) == [
        "gaussian_constraint_combined",
        "poisson_constraint_combined",
    ]


def test_cli_contrib_public_api():
    assert dir(pyhf.cli.contrib) == ["download"]


def test_contrib_utils_public_api():
    assert dir(pyhf.contrib.utils) == ["download"]


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
    assert dir(pyhf.infer.intervals) == ["upper_limits.upper_limit"]


def test_infer_intervals_upper_limit_public_api():
    assert dir(pyhf.infer.intervals.upper_limits) == [
        "linear_grid_scan",
        "toms748_scan",
        "upper_limit",
    ]


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
        "histfactory_set",
        "histosys",
        "histosys_builder",
        "histosys_combined",
        "lumi",
        "lumi_builder",
        "lumi_combined",
        "normfactor",
        "normfactor_builder",
        "normfactor_combined",
        "normsys",
        "normsys_builder",
        "normsys_combined",
        "shapefactor",
        "shapefactor_builder",
        "shapefactor_combined",
        "shapesys",
        "shapesys_builder",
        "shapesys_combined",
        "staterror",
        "staterror_builder",
        "staterror_combined",
    ]


def test_parameters_public_api():
    assert dir(pyhf.parameters) == [
        "ParamViewer",
        "constrained_by_normal",
        "constrained_by_poisson",
        "paramset",
        "reduce_paramsets_requirements",
        "unconstrained",
    ]


def test_parameters_paramsets_public_api():
    assert dir(pyhf.parameters.paramsets) == [
        "constrained_by_normal",
        "constrained_by_poisson",
        "constrained_paramset",
        "paramset",
        "unconstrained",
    ]


def test_parameters_paramview_public_api():
    assert dir(pyhf.parameters.paramview) == ["ParamViewer"]


def test_parameters_utils_public_api():
    assert dir(pyhf.parameters.utils) == ["reduce_paramsets_requirements"]


def test_patchset_public_api():
    assert dir(pyhf.patchset) == ["Patch", "PatchSet"]


def test_pdf_public_api():
    assert dir(pyhf.pdf) == ["Model", "_ModelConfig"]


def test_probability_public_api():
    assert dir(pyhf.probability) == ["Independent", "Normal", "Poisson", "Simultaneous"]


def test_readxml_public_api():
    assert dir(pyhf.readxml) == [
        "clear_filecache",
        "dedupe_parameters",
        "extract_error",
        "import_root_histogram",
        "parse",
        "process_channel",
        "process_data",
        "process_measurements",
        "process_sample",
    ]


def test_simplemodels_public_api():
    assert dir(pyhf.simplemodels) == [
        "correlated_background",
        "uncorrelated_background",
    ]


def test_utils_public_api():
    assert dir(pyhf.utils) == [
        "EqDelimStringParamType",
        "citation",
        "digest",
        "options_from_eqdelimstring",
    ]


def test_schema_public_api():
    assert dir(pyhf.schema) == [
        "load_schema",
        "path",
        "upgrade_patchset",
        "upgrade_workspace",
        "validate",
        "version",
    ]


def test_workspace_public_api():
    assert dir(pyhf.workspace) == ["Workspace"]


def test_writexml_public_api():
    assert dir(pyhf.writexml) == [
        "build_channel",
        "build_data",
        "build_measurement",
        "build_modifier",
        "build_sample",
        "indent",
    ]
