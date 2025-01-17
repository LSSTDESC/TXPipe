from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile
import numpy as np

class TXFocalPlanePlot(PipelineStage):
    """
    Make diagnostic plot of  mean measured ellipticity and residual ellipticity 
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
        import matplotlib.pyplot as plt
        
        matplotlib.use("agg")

        
        fov_x, fov_y, e1, e2, de1, de2 = self.load_stars()
        
  
        fig = self.open_output("focalplane_g", wrapper=True)
        
        weights = [e1,de1,e2,de2]
        labels = ['e1','res','e2','res2']
        
        plot, axes = plt.subplots(nrows=2, ncols=2)
        for ax,w,l in zip(axes.flat,weights,labels):
            H,xedge,yedge= np.histogram2d(fov_x, fov_y, bins=(100,100), weights=w)
            im = ax.imshow(H,cmap='RdBu')
            ax.set_ylabel(l)

        plot.subplots_adjust(right=0.85)
        cbar_ax = plot.add_axes([0.9, 0.15, 0.05, 0.7])
        plot.colorbar(im,cax=cbar_ax,cmap='RdBu')
        plt.tight_layout()
    
        fig.close()    
        
        
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