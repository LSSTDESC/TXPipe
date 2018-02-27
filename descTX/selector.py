from pipette import PipelineStage
from descTX.types import ShearCatFile, TomoCatFile
from pipette.types import YamlFile

def select(data, cuts, variant):
    n = len(data)

    s2n_cut = cuts['T']
    T_cut = cuts['s2n']
    
    s2n_col = 'mcal_T' + variant
    T_col = 'mcal_s2n_r' + variant

    s2n = data[s2n_col]
    T = data[T_col]
    Tpsf = data['mcal_Tpsf']
    flag = data['mcal_flags']

    sel  = flag==0
    sel &= (T/Tpsf)>T_cut
    sel &= s2n>s2n_cut

    return sel





class TXSelector(PipelineStage):
    name='TXSelector'
    inputs = [
        ('shear_catalog', ShearCatFile),
        ('selector_config', YamlFile),
    ]
    outputs = [
        ('tomography_catalog', TomoCatFile)
    ]

    def run(self):
        import numpy as np

        cuts, info = self.read_config()
        data = self.load_cat(info)

        sel_00 = select(data, cuts, '')
        sel_1p = select(data, cuts, '_1p')
        sel_2p = select(data, cuts, '_2p')
        sel_1m = select(data, cuts, '_1m')
        sel_2m = select(data, cuts, '_2m')

        g = data['mcal_g']
        delta_gamma = info['delta_gamma']

        R_1 = (data['mcal_g_1p'] - data['mcal_g_1m']).mean(axis=0) / delta_gamma
        R_2 = (data['mcal_g_2p'] - data['mcal_g_2m']).mean(axis=0) / delta_gamma

        S_1 = (g[sel_1p].mean() - g[sel_1m].mean()) / delta_gamma
        S_2 = (g[sel_2p].mean() - g[sel_2m].mean()) / delta_gamma

        R_1 += S_1
        R_2 += S_2

        tomo_bin = np.zeros(len(data))

        self.save_cat(tomo_bin, R_1, R_2)


    def save_cat(self, tomo_bin, R_1, R_2):
        import h5py
        filename = self.get_output('tomography_catalog')
        f = h5py.File(filename, 'w')
        n = len(tomo_bin)
        dset = f.create_dataset("tomo_bin", (n,), dtype='i')
        dset[:] = tomo_bin
        dset.attrs['R_1'] = R_1
        dset.attrs['R_2'] = R_2
        f.close()


    def choose_cuts(self):
        import yaml
        selector_config = self.get_input('selector_config')
        config = yaml.load(open(selector_config))
        
        cuts = {
            'T': config['T_cut']
            's2n': config['s2n_cut']
        }

        info = {
            'delta_gamma': config['delta_gamma']
            'nrows': config.get('nrows', 0)
        }
        return cuts, info


    def load_cat(self, info):
        import fitsio
        shear_catalog = self.get_input('shear_catalog')
        f = fitsio.FITS(shear_catalog)
        cols = ['mcal_T', 'mcal_s2n_r', 'mcal_g']
        for c in cols[:]:
            cols.append(c + "_1p")
            cols.append(c + "_1m")
            cols.append(c + "_2p")
            cols.append(c + "_2m")
        cols += ['mcal_flags', 'mcal_Tpsf']
        ext = f[1]

        if info['nrows']:
            data = ext.read_columns(cols, rows=range(info['nrows']))
        else:
            data = ext.read_columns(cols)
        f.close()
        return data


if __name__ == '__main__':
    PipelineStage.main()
