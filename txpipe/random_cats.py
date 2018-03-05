from pipette import PipelineStage
from pipette.types import YamlFile
from txpipe.dtypes import *


class TXRandomCat(PipelineStage):
    name='TXRandomCat'
    inputs = [
        ('diagnostic_maps', DiagnosticMapsFile),
        ('config', YamlFile),
    ]
    outputs = [
        ('random_cats', RandomCatFile),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
