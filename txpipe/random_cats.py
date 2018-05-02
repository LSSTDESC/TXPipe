from ceci import PipelineStage
from descformats.tx import DiagnosticMaps, YamlFile, RandomsCatalog


class TXRandomCat(PipelineStage):
    name='TXRandomCat'
    inputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]
    outputs = [
        ('random_cats', RandomsCatalog),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
