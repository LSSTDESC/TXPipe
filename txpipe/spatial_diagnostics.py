from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile
import numpy as np

class TXFocalPlanePlot(PipelineStage):
    """
    Make diagnostic plot of 
    mean measured ellipticity and residual
    as a function of position in the focal plane
    """

    name = "TXFocalPlanePlot"

    inputs = [
        ("star_catalog", HDFFile),
    ]

    outputs = [
        ("focalplane_g", PNGFile),
    ]

    config_options = {}

    def run(self):
        import matplotlib
        matplotlib.use("agg")
                
        self.plot()
        
        
    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g = f["stars"]
            fov_x = g["fov_x"][:]
            fov_y = g["fov_y"][:]
            e1 = g["measured_e1"][:]
            e2 = g["measured_e2"][:]
            de1 = e1 - g["model_e1"][:]
            de2 = e2 - g["model_e2"][:]
            
        return fov_x, fov_y, e1, e2, de1, de2
    
    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import TwoSlopeNorm

        fov_x, fov_y, e1, e2, de1, de2 = self.load_stars()

        fig = self.open_output("focalplane_g", wrapper=True, figsize=(10,10))
        
        weights = [e1,  e2, de1*10, de2*10]
        labels = [r'e$_1$',r'e$_2$',r'res $[\times 10]$',r'res $[\times 10]$']
        
        cmap=cm.seismic
        
        Hnorm = []
        for w in weights:
            Hw, _, _= np.histogram2d(fov_x, fov_y, bins=(100,100), weights=w)
            H, _, _= np.histogram2d(fov_x, fov_y, bins=(100,100))
            Hnorm.append(Hw/H)
     
        divnorm=TwoSlopeNorm(vmin=-0.1, vcenter=0., vmax=0.1)
        
        for i,l in zip(range(0,4), labels):
            plt.subplot(2,2,i+1)
            im = plt.imshow(Hnorm[i], cmap=cmap,norm=divnorm)
            plt.ylabel(l)
            plt.xticks([])
            plt.yticks([])
            
        plt.subplots_adjust(left=0.1,right=0.7)
        cax = plt.axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(im,cax)      
        fig.close()