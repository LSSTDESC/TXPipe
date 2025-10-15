from .base_stage import PipelineStage
from .data_types import HDFFile
import numpy as np
from ceci.config import StageParameter


class TXExposureInfo(PipelineStage):
    """
    Ingest exposure information from an OpSim database

    This is used later for measurements, e.g. shear around exposure centers.
    """

    name = "TXExposureInfo"
    parallel = False
    inputs = []
    outputs = [
        ("exposures", HDFFile),
    ]
    config_options = {
        "dc2_name": StageParameter(str, "1.2p", msg="Name of the DC2 run to use."),
        "opsim_db": StageParameter(str, "/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db", msg="Path to the opsim database file."),
        "propId": StageParameter(int, 54, msg="Proposal ID to filter visits."),
    }

    def run(self):
        from astropy.io import fits
        import sqlite3

        # find butler by name using this tool.
        # change later to general repo path.
        from desc_dc2_dm_data import get_butler
        from lsst.daf.persistence import NoResults

        run = self.config["dc2_name"]
        propId = self.config["propId"]

        print(f"Getting butler for repo {run}")
        butler = get_butler(run)
        print("Butler loaded")

        # central detector in whole focal plane, just as example
        refs = butler.subset("calexp", raftName="R22", detectorName="S11")
        n = len(refs)
        print(f"Found {n} exposure centers.  Reading exposure info.")

        matching_visits = self.find_matching_opsim_visits()
        nmatch = len(matching_visits)
        print(f"Found list of {nmatch} visits with propId=={propId}")

        float_params = [
            "mjd-obs",
            "bore-ra",
            "bore-dec",
            "bore-az",
            "bore-alt",
            "bore-rotang",
            "bore-airmass",
            "humidity",
            "bgmean",
            "bgvar",
            "magzero_rms",
            "colorterm1",
            "colorterm2",
            "colorterm3",
            "exptime",
            "darktime",
        ]

        int_params = [
            "expid",
            "magzero_nobj",
            "ap_corr_map_id",
            "psf_id",
            "skywcs_id",
        ]

        str_params = [
            "date-avg",
            "timesys",
            "rottype",
            "filter",
            "testtype",
            "obstype",
        ]
        # We can add WCS information here if needed, there is lots
        # of it.

        params = float_params + int_params + str_params
        # Spaces for output columns
        data = {p: list() for p in params}

        num_params = float_params + int_params
        # Loop through the images and get their metadata
        for i, ref in enumerate(refs):
            # Progress update
            if i % 100 == 0:
                print(f"Reading metadata for exposure {i+1} / {n}")

            # Read the metadata for this exposure reference
            try:
                metadata = butler.get("calexp_md", dataId=ref.dataId).toDict()
            except NoResults:
                continue
            obsid = int(str(metadata["EXPID"])[:-3])
            if obsid not in matching_visits:
                continue

            # columns that we want to save to file, from the metadata
            for p in params:
                data[p].append(metadata[p.upper()])

        m = len(data["bore-ra"])
        f = 100.0 * m / n
        print(f"{m} / {n} visits match propId={propId} ({f:.2f}%)")

        # Save output
        f = self.open_output("exposures")
        g = f.create_group("exposures")

        for name in num_params:
            g.create_dataset(name, data=data[name])

        for name in str_params:
            # H5PY cannot deal with fixed-length unicode arrays (the numpy default in py3)
            # so convert to ASCII
            g.create_dataset(name, data=np.array(data[name], dtype="S"))

        f.close()

    def find_matching_opsim_visits(self):
        import sqlite3

        db = self.config["opsim_db"]
        propId = self.config["propId"]
        connection = sqlite3.connect(db)
        cursor = connection.cursor()
        cursor.execute(
            "select obsHistID from summary where propId=:propId", {"propId": propId}
        )
        obsHistID = {Id[0] for Id in cursor.fetchall()}
        return obsHistID
