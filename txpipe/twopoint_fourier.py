from ceci import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('random_catalog', RandomsCatalog),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]

    def run(self):
        pass

class TXTwoPointFourier(PipelineStage):
    name='TXTwoPointFourier'
    inputs = [
        ('shear_catalog',MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('mask',DiagnosticMap),
        ('syst',DiagnosticMap),
        ('config',YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    def run(self) :
        config = self.read_config()

        zbin_edges = config['zbin_edges']
        zbins = list(zip(zbin_edges[:-1], zbin_edges[1:]))
        nbins = len(zbins)

        print('number of bins', nbins)
        



    
if __name__ == '__main__':
    PipelineStage.main()
