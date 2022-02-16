Stages requiring the LSST environment
-------------------------------------

Some ingestion stages may need the LSST environment.


1. Start the LSST env using `python /global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-stack` (or any new instructions from https://confluence.slac.stanford.edu/display/LSSTDESC/Setting+Up+the+DM+Stack+at+NERSC)

2. Create and start a virtual env based on this (first time only)

```
python -m venv --system-site-packages lsst-env
. lsst-env/bin/activate
```

3. Install ceci in that venv

```
pip install git+https://github.com/LSSTDESC/ceci@v0.7
```

Then you're ready.
