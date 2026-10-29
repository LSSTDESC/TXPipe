import pytest

pytest.importorskip("smokescreen")
pytest.importorskip("firecrown")

import numpy as np
import pyccl
import sacc
import yaml

from ..smokescreen_blinding import SmokescreenBlinding
from ..utils.theory import theory_3x2pt, make_bias_parameters
from .test_firecrown import make_sacc

REFERENCE_COSMOLOGY = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96,
    "sigma8": 0.81,
}
LENS_BIAS = {"lens_0": 1.2, "lens_1": 1.4, "lens_2": 1.6}
# All the parameters the TXPipe theory model needs, with a constant bias per bin
SYSTEMATICS = make_bias_parameters(LENS_BIAS, make_sacc(), None)


def make_data(space):
    """
    Make a small 3x2pt sacc file with the TXPipe tracer pairs.

    The values do not matter for blinding, which adds a shift to them,
    so we use distinct non-zero numbers. In harmonic space we also add
    B-mode points, which the theory models cannot predict.
    """
    sacc_data = make_sacc()
    sources = [t for t in sacc_data.tracers if t.startswith("source")]
    lenses = [t for t in sacc_data.tracers if t.startswith("lens")]

    if space == "real":
        tag = "theta"
        values = np.geomspace(5.0, 100.0, 4)
        shear_types = ["galaxy_shear_xi_plus", "galaxy_shear_xi_minus"]
        cross_type = "galaxy_shearDensity_xi_t"
        density_type = "galaxy_density_xi"
    else:
        tag = "ell"
        values = np.geomspace(20.0, 1000.0, 4)
        shear_types = ["galaxy_shear_cl_ee", "galaxy_shear_cl_bb"]
        cross_type = "galaxy_shearDensity_cl_e"
        density_type = "galaxy_density_cl"

    pairs = []
    for dt in shear_types:
        for i, s1 in enumerate(sources):
            for s2 in sources[: i + 1]:
                pairs.append((dt, s2, s1))
    for s in sources:
        for l in lenses:
            pairs.append((cross_type, s, l))
    for l in lenses:
        pairs.append((density_type, l, l))

    for dt, t1, t2 in pairs:
        for x in values:
            value = 1e-6 * (len(sacc_data) + 1)
            sacc_data.add_data_point(dt, (t1, t2), value, **{tag: x})

    sacc_data.add_covariance(1e-12 * np.ones(len(sacc_data)))
    return sacc_data


def run_stage(tmp_path, sacc_data, name="blinded", **config):
    """
    Save the data, then run the blinding stage on it as a pipeline would.
    Returns the paths to the raw and blinded files.
    """
    raw_path = str(tmp_path / "raw.sacc")
    if sacc_data is not None:
        sacc_data.save_fits(raw_path, overwrite=True)
    blinded_path = str(tmp_path / f"{name}.sacc")

    options = {
        "reference_cosmology": REFERENCE_COSMOLOGY,
        "shifts_dict": {"sigma8": [0.75, 0.85], "w0": [-1.2, -0.8]},
        "systematics": SYSTEMATICS,
    }
    options.update(config)
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump({"SmokescreenBlinding": options}, f)

    stage = SmokescreenBlinding(
        {
            "twopoint_data_raw": raw_path,
            "twopoint_data_blinded": blinded_path,
            "config": str(config_path),
        }
    )
    stage.run()
    stage.finalize()
    return raw_path, blinded_path


def test_blinding_real_space(tmp_path):
    raw, blinded = run_stage(tmp_path, make_data("real"))
    raw = sacc.Sacc.load_fits(raw)
    blinded = sacc.Sacc.load_fits(blinded)

    # Every point is modelled, so every point should be shifted
    assert np.all(blinded.mean != raw.mean)
    assert np.all(np.isfinite(blinded.mean))

    # Everything else is unchanged
    assert list(blinded.tracers) == list(raw.tracers)
    np.testing.assert_array_equal(blinded.covariance.dense, raw.covariance.dense)
    for d_raw, d_blinded in zip(raw.data, blinded.data):
        assert d_blinded.data_type == d_raw.data_type
        assert d_blinded.tracers == d_raw.tracers
        assert d_blinded.tags == d_raw.tags

    assert blinded.metadata["concealed"]
    assert blinded.metadata["blinding_method"] == "smokescreen"
    assert blinded.metadata["blinding_likelihood"] == "theory_model.py"
    assert "smokescreen_version" in blinded.metadata


def test_blinding_seed(tmp_path):
    sacc_data = make_data("real")
    _, first = run_stage(tmp_path, sacc_data, name="first", seed="1234")
    _, second = run_stage(tmp_path, sacc_data, name="second", seed="1234")
    _, other = run_stage(tmp_path, sacc_data, name="other", seed="4321")

    first = sacc.Sacc.load_fits(first).mean
    second = sacc.Sacc.load_fits(second).mean
    other = sacc.Sacc.load_fits(other).mean

    np.testing.assert_array_equal(first, second)
    assert np.all(first != other)


def test_blinding_shift_is_theory_difference(tmp_path):
    # With a fixed shift the blinded data should be the raw data plus
    # the difference between the theory at the shifted and reference cosmology.
    sacc_data = make_data("real")
    raw, blinded = run_stage(tmp_path, sacc_data, shifts_dict={"sigma8": 0.9})
    delta = sacc.Sacc.load_fits(blinded).mean - sacc.Sacc.load_fits(raw).mean

    reference = pyccl.Cosmology(**REFERENCE_COSMOLOGY)
    shifted = pyccl.Cosmology(**{**REFERENCE_COSMOLOGY, "sigma8": 0.9})
    expected = theory_3x2pt(shifted, sacc_data, bias=LENS_BIAS).mean - theory_3x2pt(reference, sacc_data, bias=LENS_BIAS).mean

    np.testing.assert_allclose(delta, expected, rtol=1e-10)


def test_blinding_encrypt_unblinded(tmp_path):
    from smokescreen.encryption import decrypt_file

    encryption_dir = tmp_path / "encrypted"
    encryption_dir.mkdir()
    raw, _ = run_stage(tmp_path, make_data("real"), encrypt_unblinded=True, encryption_dir=str(encryption_dir))

    # The original is kept, and the encrypted copy decrypts back to it
    encrypted = encryption_dir / "raw.encrpt"
    key = encryption_dir / "raw.key"
    assert encrypted.exists() and key.exists()
    with open(raw, "rb") as f:
        assert decrypt_file(str(encrypted), str(key)) == f.read()


def test_blinding_delete_unblinded(tmp_path):
    raw, _ = run_stage(tmp_path, make_data("real"), delete_unblinded=True)
    with open(raw) as f:
        assert f.read() == ""

    # Running again on the emptied file should say what went wrong
    with pytest.raises(ValueError, match="zero size"):
        run_stage(tmp_path, None, name="again")


def test_blinding_harmonic_space_b_modes(tmp_path):
    # The theory model cannot predict B-modes, so they should be left as they are
    raw, blinded = run_stage(tmp_path, make_data("fourier"))
    raw = sacc.Sacc.load_fits(raw)
    blinded = sacc.Sacc.load_fits(blinded)

    b_modes = raw.indices("galaxy_shear_cl_bb")
    modelled = np.setdiff1d(np.arange(len(raw)), b_modes)
    assert len(b_modes) > 0
    np.testing.assert_array_equal(blinded.mean[b_modes], raw.mean[b_modes])
    assert np.all(blinded.mean[modelled] != raw.mean[modelled])


# A minimal firecrown likelihood for xi_- only. It is used to test custom
# likelihoods, and if REMOVE is set it removes the xi_+ points from the
# sacc data instead of just not using them.
SHEAR_LIKELIHOOD = """
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood import TwoPoint, ConstGaussian, NamedParameters
from firecrown.modeling_tools import ModelingTools

REMOVE = {remove}

def build_likelihood(build_parameters):
    sacc_data = build_parameters.data["sacc_data"]
    if REMOVE:
        sacc_data.remove_selection(data_type="galaxy_shear_xi_plus")
    data_type = "galaxy_shear_xi_minus"
    sources = {{t: wl.WeakLensing(sacc_tracer=t) for t in sacc_data.tracers if t.startswith("source")}}
    statistics = [
        TwoPoint(source0=sources[t1], source1=sources[t2], sacc_data_type=data_type)
        for t1, t2 in sacc_data.get_tracer_combinations(data_type)
    ]
    likelihood = ConstGaussian(statistics=statistics)
    likelihood.read(sacc_data)
    return likelihood, ModelingTools()
"""


def write_likelihood(tmp_path, remove=False):
    path = tmp_path / "shear_likelihood.py"
    path.write_text(SHEAR_LIKELIHOOD.format(remove=remove))
    return str(path)


def test_blinding_custom_likelihood(tmp_path):
    likelihood_path = write_likelihood(tmp_path)
    raw, blinded = run_stage(tmp_path, make_data("real"), likelihood_path=likelihood_path, systematics={})
    raw = sacc.Sacc.load_fits(raw)
    blinded = sacc.Sacc.load_fits(blinded)

    # Only the points the likelihood models are blinded
    xim = raw.indices("galaxy_shear_xi_minus")
    others = np.setdiff1d(np.arange(len(raw)), xim)
    assert np.all(blinded.mean[xim] != raw.mean[xim])
    np.testing.assert_array_equal(blinded.mean[others], raw.mean[others])
    assert blinded.metadata["blinding_likelihood"] == "shear_likelihood.py"


def test_blinding_likelihood_removing_points(tmp_path):
    likelihood_path = write_likelihood(tmp_path, remove=True)
    with pytest.raises(ValueError, match="does not match the input data"):
        run_stage(tmp_path, make_data("real"), likelihood_path=likelihood_path, systematics={})


def test_blinding_missing_likelihood(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_stage(tmp_path, make_data("real"), likelihood_path=str(tmp_path / "missing.py"))


def test_blinding_no_shifts(tmp_path):
    with pytest.raises(ValueError, match="No shifts_dict"):
        run_stage(tmp_path, make_data("real"), shifts_dict={})
