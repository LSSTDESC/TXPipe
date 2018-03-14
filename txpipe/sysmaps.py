from pipette import PipelineStage
from descformats.tx import MetacalCatalog, DiagnosticMaps, YamlFile


class TXDiagnosticMaps(PipelineStage):
    name='TXDiagnosticMaps'
    inputs = [
        ('shear_cat', MetacalCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
