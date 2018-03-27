from ceci import PipelineStage

class shearMeasurementPipe(PipelineStage):
    name='shearMeasurementPipe'
    inputs = ['DM']
    outputs = ['shear_catalog']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    shearMeasurementPipe reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    shearMeasurementPipe writing to {filename}")
            open(filename,'w').write("shearMeasurementPipe was here \n")


class PZEstimationPipe(PipelineStage):
    name='PZEstimationPipe'
    inputs = ['DM','fiducial_cosmology']
    outputs = ['photoz_pdfs']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    PZEstimationPipe reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    PZEstimationPipe writing to {filename}")
            open(filename,'w').write("PZEstimationPipe was here \n")


class WLGCRandoms(PipelineStage):
    name='WLGCRandoms'
    inputs = ['diagnostic_maps']
    outputs = ['random_catalog']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    WLGCRandoms reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    WLGCRandoms writing to {filename}")
            open(filename,'w').write("WLGCRandoms was here \n")




class SourceSummarizer(PipelineStage):
    name='SourceSummarizer'
    inputs = ['tomography_catalog','photoz_pdfs','diagnostic_maps']
    outputs = ['source_summary_data']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    SourceSummarizer reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    SourceSummarizer writing to {filename}")
            open(filename,'w').write("SourceSummarizer was here \n")


class SysMapMaker(PipelineStage):
    name='SysMapMaker'
    inputs = ['DM']
    outputs = ['diagnostic_maps']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    SysMapMaker reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    SysMapMaker writing to {filename}")
            open(filename,'w').write("SysMapMaker was here \n")


class WLGCTwoPoint(PipelineStage):
    name='WLGCTwoPoint'
    inputs = ['tomography_catalog','shear_catalog','diagnostic_maps','random_catalog']
    outputs = ['twopoint_data']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    WLGCTwoPoint reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    WLGCTwoPoint writing to {filename}")
            open(filename,'w').write("WLGCTwoPoint was here \n")


class WLGCCov(PipelineStage):
    name='WLGCCov'
    inputs = ['fiducial_cosmology','tomography_catalog','shear_catalog','source_summary_data','diagnostic_maps']
    outputs = ['covariance']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    WLGCCov reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    WLGCCov writing to {filename}")
            open(filename,'w').write("WLGCCov was here \n")


class WLGCSummaryStatistic(PipelineStage):
    name='WLGCSummaryStatistic'
    inputs = ['twopoint_data','covariance','source_summary_data']
    outputs = ['wlgc_summary_data']
    
    def run(self):
        for inp in self.inputs:
            filename = self.get_input(inp)
            print(f"    WLGCSummaryStatistic reading from {filename}")
            open(filename)

        for out in self.outputs:
            filename = self.get_output(out)
            print(f"    WLGCSummaryStatistic writing to {filename}")
            open(filename,'w').write("WLGCSummaryStatistic was here \n")


if __name__ == '__main__':
    cls = PipelineStage.main()