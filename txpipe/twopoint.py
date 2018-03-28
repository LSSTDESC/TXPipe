from ceci import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('random_catalog', RandomsCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]

    def run(self):
        pass



if __name__ == '__main__':
    PipelineStage.main()
