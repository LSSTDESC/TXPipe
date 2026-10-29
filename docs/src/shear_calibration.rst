Shear calibration guide
=======================

Raw cosmic shear values in catalogs cannot be used directly for cosmology; they are subject to selection and measurement biases that must be calibrated out first. It is not typically possible to do this calibration individually on each object; instead an ensemble of objects must be chosen, and the calibration for that ensemble applied to all the objects in it.

In our primary use case one ensemble is one tomographic bin, but for shear diagnostics we also have to calibrate different data sub-samples. For example, if we want to measure the shear in bins of signal-to-noise, then one ensemble is all objects in a given signal-to-noise bin. The calibration factor we find will be different to the tomographic calibration factors.

How calibration fits into the pipeline
--------------------------------------

Calibration is the last of three steps that turn a shear catalog into a calibrated tomographic sample: :ref:`selection, tomography and calibration<Source selection and tomography>`. The ``TXComputeShearCalibration*`` stages (one per catalog type, in ``txpipe/shear_calibration/``) take the cuts from the selection stage and the bin assignments from the tomography stage, and use the calculator classes described below to find the calibration of each tomographic bin. They save the results in the "response" and "counts" groups of the :ref:`shear tomography catalog<Shear tomography catalogs>`.

Later stages load those values with the calibrator classes. The ``TXShearCalibration`` stage (in ``txpipe/calibrate.py``) uses them to make calibrated tomographic catalogs, and ``TXSourceMaps`` turns those into tomographic maps.

If you are doing straightforward analysis on tomographic quantities you may be able to avoid understanding the shear calibration at all, since these stages do it all for you. The classes below are for people who want to write new diagnostics, or add a new type of shear catalog.

TXPipe uses several classes in shear calibration, all in the ``txpipe.calibration_utils`` sub-module.

- ``CalibrationCalculator``: the calculator sub-classes compute the shear calibration factor.
- ``Calibrator``: the calibrator sub-classes load the shear calibration factor from a file and then apply it to a sample.
- ``MeanShearInBins``: this class calculates and applies the shear calibration factors for the specific case where you are binning the shear catalog in bins of another quantity, such as size or signal-to-noise.

Calculator Classes
------------------

The calculator classes can be found in ``txpipe/calibration_utils/calibration_calculators.py``. There is one for each shear calibration method (``MetacalCalculator``, ``MetaDetectCalculator``, ``ScalarMetaDetectCalculator``, ``LensfitCalculator``, ``HSCCalculator``, and ``MockCalculator`` for catalogs that need no calibration), and ``CalibrationCalculator.create_calculator`` chooses the right one from the catalog type. Their main use is in the ``TXComputeShearCalibration*`` stages in ``txpipe/shear_calibration/``, and for diagnostics like ``TXResponseInBins``. Their output is saved into the :ref:`tomography file<Shear tomography catalogs>` "response" group.

The calculator classes are designed to operate on streamed data (i.e. they do not load the entire catalog at once, but a chunk at a time).  The expected life cycle of a calculator class is:

1. Instantiate the calculator class with a `selector` function (see below) and other options.
2. Call the `add_data` method repeatedly with chunks of data (can be done in parallel).
3. Call the `collect` method at the end to get the final calibration factors.

The **selector** function is used because in some cases we need to calculate a selection bias quantity, and to do this we run the selection on different data variants. It should take this form::

    def selector(data: dict[str,np.ndarray], *args, **kwargs):
        ...
        return selection

The `data` argument is a dictionary of numpy arrays, where the keys are the column names and the values are the data. The function should return something we can use to index the data, such as a boolean array or an integer array.  The extra arguments are passed in from the `add_data` method, and can be anything used in the selection, such the index of the tomographic bin being selected. They will be passed in when calling `add_data`.

In the calibration stages the selector combines the object's tomographic bin assignment and its selection flag, in the stage methods ``select`` and ``select_any_bin``.

The **add_data** method should be called with each chunk of data loaded in one-by-one. Its `data` argument is the same as the `selector` function, and the `args` and `kwargs` are passed to the `selector` function.

For metadetect catalogs the variants ("ns", "1p", ...) are separate catalogs of different lengths, and so `data` has keys like ``"1p/g1"`` instead of ``"g1_1p"``. For these calculators you can call `add_data` with all the variants in one chunk, or with a chunk containing only some of them: variants that are missing from a chunk are skipped. This allows the stages to work through one variant at a time.

The calculator classes return an instance of `BinStats`, a little class that just contains statistics about the bin (means, variances, counts, etc), and, most importantly, an instance of a `Calibrator` subclass that actually does the calibration.

Calibrator Classes
------------------

The calibrator classes read the data from the tomography file response group and apply it to shear catalogs, maps, or similar values. They can be found in ``txpipe/calibration_utils/calibrators.py``. The primary place they are used is to generate calibrated tomographic catalogs in the ``TXShearCalibration`` stage in ``calibrate.py``.

The base class, `Calibrator`, has a `load` method which takes the name of the tomography file to load from, and then determines which shear catalog type it is and returns a `Calibrator` sub-class appropriate for that type. This method also takes a "null" keyword which you can use to tell it to ignore the calibration factors and return a dummy calibrator that does nothing. This is useful when you have true shear values for testing.

All the subclasses then have an `apply` method which takes `g1` and `g2` arrays/floats and returns calibrated values. By default a mean shear value is also subtracted. They have several other methods that calibrate variances etc.

The calibration for each type of catalog is different:

.. list-table::
    :header-rows: 1

    * - Catalog type
      - Calibrator
      - Calibration
    * - Metacal
      - ``MetaCalibrator``
      - 2x2 matrix, the sum of a shear response (measured per object and averaged over the bin) and a selection response (measured by repeating the selection on the sheared variants)
    * - Metadetect
      - ``MetaDetectCalibrator``
      - 2x2 matrix; the response is measured from the sheared catalogs, so the selection response is included automatically, and is only available per bin
    * - Scalar metadetect
      - ``ScalarMetaDetectCalibrator``
      - A single number, assuming the response matrix is a multiple of the identity
    * - Lensfit
      - ``LensfitCalibrator``
      - A multiplicative factor K for each bin, and additive terms C that differ between the north and south of the survey
    * - HSC
      - ``HSCCalibrator``
      - Multiplicative and additive terms, with a per-object response
    * - Simple
      - ``NullCalibrator``
      - None; the shears are used as they are

MeanShearInBins
---------------

The `MeanShearInBins` class is designed to simplify null tests in which the the mean shear is calculated in different bins of some quantity, such as signal-to-noise or size. It calculates the shear calibration factors for each bin and applies them to the mean shear in that bin. It is used for null testing in ``TXSourceDiagnosticPlots``.
