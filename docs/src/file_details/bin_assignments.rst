Tomographic bin assignments
- **Simple, lensfit and HSC catalogs:** a single dataset ``bin``.
- **Metacal catalogs:** datasets ``bin``, ``bin_1p``, ``bin_1m``, ``bin_2p`` and ``bin_2m``, for the unsheared and sheared variants.
- **Metadetect catalogs:** a dataset ``bin`` in a group for each variant, so ``ns/bin``, ``1p/bin``, ``1m/bin``, ``2p/bin`` and ``2m/bin``.
- **Scalar metadetect catalogs:** the same, for the variants ``ns``, ``1p`` and ``1m`` only.

All are 16-bit integers.

The metacal and metadetect sheared-variant assignments are needed to compute the selection response. For metacal only the unsheared assignment ends up in the final tomography catalog.
