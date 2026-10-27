with open("examples/dp2/analysis1.1.yml", "r") as fin:
    pipeline = fin.read()

with open("examples/dp2/config.yml", "r") as fin:
    config = fin.read()
    

# /global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz/.gen_paths/txpipe_dp2_shear_catalog_desc_metaDetect_1deg2_0.3.0/mini_desc_dp2_shear_catalog.hdf5
# /global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz/.gen_paths/txpipe_dp2_shear_catalog_desc_metaDetect_mask_1deg2_0.3.0/mini_mask.hdf5

pipeline = pipeline.replace("nprocess: 32", "nprocess: 8")
pipeline = pipeline.replace("nprocess: 64", "nprocess: 8")
pipeline = pipeline.replace("nprocess: 128", "nprocess: 8")
pipeline = pipeline.replace("threads_per_process: 16", "threads_per_process: 8")
pipeline = pipeline.replace("threads_per_process: 32", "threads_per_process: 8")
pipeline = pipeline.replace("threads_per_process: 128", "threads_per_process: 8")
pipeline = pipeline.replace("config: examples/dp2/config.yml", "config: examples/dp2/config-mini.yml")
pipeline = pipeline.replace("mask:\n      id: 148", "mask: /global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz/.gen_paths/txpipe_dp2_shear_catalog_desc_metaDetect_mask_1deg2_0.3.0/mini_mask.hdf5")
pipeline = pipeline.replace("shear_catalog:\n        id: 147", "shear_catalog: /global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz/.gen_paths/txpipe_dp2_shear_catalog_desc_metaDetect_1deg2_0.3.0/mini_desc_dp2_shear_catalog.hdf5")
pipeline = pipeline.replace("registry: {}", "#registry: {}")
pipeline = pipeline.replace("name: nersc-interactive", "name: local")
pipeline = pipeline.replace("""classname: SOMPZEstimatorDeep
    nprocess: 8
    threads_per_process: 2""", """classname: SOMPZEstimatorDeep
    nprocess: 2
    
    threads_per_process: 2""")

config = config.replace("chunk_size: 10000\n", "chunk_size: 1000\n")
config = config.replace("chunk_size: 100000", "chunk_size: 10000")
config = config.replace("chunk_size: 1000000", "chunk_size: 100000")

with open("examples/dp2/analysis1.1-mini.yml", "w") as fout:
    fout.write(pipeline)


with open("examples/dp2/config-mini.yml", "w") as fout:
    fout.write(config)
