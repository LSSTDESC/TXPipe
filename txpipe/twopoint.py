rom pipette import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        #('galaxy_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        #('random_catalog', RandomsCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]

    def run(self):

        # read in a shear catalog
	# read in tomography catalog
        # read in the different columns (e1, e2, ra, dec, metacal)
        # select one z bin
        # call treecorr
        # calculate shear-shear
	# store intermediate output 


if __name__ == '__main__':
    PipelineStage.main()
