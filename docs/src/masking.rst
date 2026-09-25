Masking Guide
=======================

Many TXPipe stages will require a mask describing the regions of the sky that contain valid data. By default, TXPipe assumes that this mask applies primarily to the lens sample and is the same for each tomographic bin.

The masks are used for many purposes withiin TXPipe including generating randoms, calculating clustering with a pixel-based estimator, and computing corrective systemtics weights.

TXPipe masks are typically saved as healsparse maps in HDF5 format. 

The mask can be: 

- A **binary** mask, where each pixel is either inside the footprint (1) or outside it (0). These masks are typically generated at high resolution
- A **fractional coverage** mask (a fracdet map), where each pixel stores the fraction of the pixel covered by valid observations. These masks can be stored at lower resolution while retaining information about survey boundaries.

Most mask-generation stages operate on a set of auxiliary survey property maps stored in the ``aux_lens_maps`` input. These maps contain quantities such as depth, exposure counts, survey property measurements, or bright-object masks that can be combined to define the final footprint.

Masking Classes
------------------

The mask-generation classes are defined in ``txpipe/masks.py``. Each stage reads one or more auxiliary maps, applies a set of threshold cuts, and writes a final mask map that can be used by downstream stages.

Currently, the primary mask-generation stages are:

- ``TXSimpleMask``: Generates a binary survey mask from cuts on ``depth`` and ```bright_objects``.
- ``TXSimpleMaskFrac``: Like ``TXSimpleMask`` but returns a fracdet of the resulting mask
- ``TXSimpleMaskSource``: Generates a separate mask for source samples. Uses the ``source_map`` to define the footprint
- ``TXCustomMask``: A flexible framework for defining survey masks using arbitrary combinations of auxiliary maps. Cuts are defined in the config file

Ingestion Classes
------------------

If you have a pre-defined mask you can input this directly to TXPipe as long as it has been saved with healsparse.

If you need to ingest a set of auxiliary maps this can be done with ``txpipe.ingest.TXIngestMapsHsp``

