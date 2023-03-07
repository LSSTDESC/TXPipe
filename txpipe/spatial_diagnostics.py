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
        import matplotlib.pyplot as plt
        
        
        fov_x, fov_y, e1, e2, de1, de2 = self.load_stars()
        
        X = plt.hist2d(fov_x, fov_y, bins=(100,100))
        
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
        
        fig = self.open_output("focalplane_g", wrapper=True)
        plt.subplot(2,2,1)
        plt.hist2d(fov_x, fov_y, bins=(100,100), weights=e1)
        plt.ylabel('mean PSF e1')
        plt.subplot(2,2,2)
        plt.hist2d(fov_x, fov_y, bins=(100,100), weights=de1)
        plt.ylabel('residual')
        plt.subplot(2,2,3)
        plt.hist2d(fov_x, fov_y, bins=(100,100), weights=e2)
        plt.ylabel('mean PSF e2')
        plt.subplot(2,2,4)
        plt.hist2d(fov_x, fov_y, bins=(100,100), weights=de2)
        plt.ylabel('residual')
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