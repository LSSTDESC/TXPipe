DESC 3x2pt Pipeline Stages
--------------------------

Run this on cori before testing:

```bash
module swap PrgEnv-intel  PrgEnv-gnu
module load python/3.6-anaconda-4.4
module load cfitsio/3.370-reentrant
source activate /global/projecta/projectdirs/lsst/groups/WL/users/zuntz/env
export PYTHONPATH=$PYTHONPATH:/global/cscratch1/sd/zuntz/pipe/pipette
```
