from .base_stage import PipelineStage
from .data_types import SACCFile


class TXNullBlinding(PipelineStage):
    """
    Pretend to blind but actually do nothing.

    This null stage trivially copies the real raw data to real without blinding,
    for use with simulated data, etc.
    """

    name = "TXNullBlinding"
    parallel = False
    inputs = [
        ("twopoint_data_real_raw", SACCFile),
    ]
    outputs = [
        ("twopoint_data_real", SACCFile),
    ]
    config_options = {}

    def run(self):
        """
        Run the analysis for this stage.

         - Load two point SACC file
         - Copy two point SACC file twopoint_data_real_raw to twopoint_data_real
        """
        import shutil

        unblinded_fname = self.get_input("twopoint_data_real_raw")
        shutil.copyfile(unblinded_fname, self.get_output("twopoint_data_real"))
