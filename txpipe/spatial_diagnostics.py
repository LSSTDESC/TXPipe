from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile, StarCatalog
import numpy as np

class TXFocalPlanePlot(PipelineStage):
    """
    Make diagnostic plot of 
    mean measured ellipticity and residual
    as a function of position in the focal plane
    """

    name = "TXFocalPlanePlot"

    inputs = [
        ("star_catalog", StarCatalog),
    ]

    outputs = [
        ("focalplane_g", PNGFile),
    ]

    config_options = {}

    def run(self):
        import matplotlib.pyplot as plt
        
        
        fov_x, fov_y, e1, e2, de1, de2 = self.load_stars()
        
        X = mplot.hist2d(fov_x, fov_y, bins=(100,100))
        
        x_edge = X[1]
        y_edge = X[2]
        
        PSF_e1_err = np.zeros((100,100))
        PSF_e2_err = np.zeros((100,100))

        for i in range(100):
            for j in range(100):
                mask = (fov_x>=x_edge[i])*(fov_x<x_edge[i+1])*(fov_y>=y_edge[j])*(fov_y<y_edge[j+1])
                if len(e1[mask])>0:
                    PSF_e1_err[i][j] = np.mean(de1[mask])
                    PSF_e2_err[i][j] = np.mean(de2[mask])
        
        f = self.open_output("focalplane_g", wrapper=True, figsize=(20,20))
            axs = plt.subplots(2,2)
            axs[0,0].hist2d(fov_x, fov_y, bins=(100,100), weights=e1)
            axs[0, 0].set_ylabel('mean PSF e1')
            axs[0,1].hist2d(fov_x, fov_y, bins=(100,100), weights=de1)
            axs[0, 0].set_ylabel('mean residual PSF e1')
            axs[1,0].hist2d(fov_x, fov_y, bins=(100,100), weights=e2)
            axs[0, 0].set_ylabel('mean PSF e2')
            axs[1,1].hist2d(fov_x, fov_y, bins=(100,100), weights=de2)
            axs[0, 0].set_ylabel('mean residual PSF e2')
        f.close()
        
        
                    
        
        
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