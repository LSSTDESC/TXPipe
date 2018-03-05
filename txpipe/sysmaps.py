from pipette import PipelineStage
from pipette.types import YamlFile
from txpipe.dtypes import *


class TXDiagnosticMaps(PipelineStage):
    name='TXDiagnosticMaps'
    inputs = [
        ('shear_cat', ShearCatFile),
        ('config', YamlFile),
    ]
    outputs = [
        ('diagnostic_maps', DiagnosticMapsFile),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
