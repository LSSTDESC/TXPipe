from pipette import PipelineStage
from pipette.types import YamlFile
from txpipe.dtypes import *


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', ShearCatFile),
        ('tomography_catalog', TomoCatFile),
        ('random_catalog', TomoCatFile),
        ('config', YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
