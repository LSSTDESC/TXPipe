Shear calibration guide
=======================

Raw cosmic shear values in catalogs cannot be used directly for cosmology; they are subject to selection and measurement biases that must be calibrated out first. It is not typically possible to do this calibration individually on each object; instead an ensemble of objects must be chosen, and the calibration for that ensemble applied to all the objects in it.

In our primary use case one ensemble is one tomographic bin, but for shear diagnostics we also have to calibrate different data sub-samples. For exampe, if we want to measure the shear in bins of signal-to-noise, for example, then one ensemble is all objects in a given signal-to-noise bin. The calibration factor we find will be different to the tomographic calibration factors.

TXPipe uses several classes in shear calibration, all in the `txpipe.utils` sub-module.

- `CalibrationCalculator`: the calculator sub-classes compute the shear calibration factor.
- `Calibrator`: the calibrator sub-classes load the shear calibration factor from a file and then apply it to a sample.
- `MeanShearInBins`: this class calculates and applies the shear calibration factors for the specific case where you are binning the shear catalog in bins of another quantity, such as size or signal-to-noise.

If you are doing straightforward analysis on tomographic quantities you may be able to avoid understanding the shear calibration at all; the `TXShearCalibration` stage generates calibrated tomographic catalogs for you, and `TXSourceMaps` turns these into tomographic maps in the `source_maps`.

Calculator Classes
------------------

The calculator classes can be found in `txpipe/utils/calibration_tools.py`. There is one for each shear calibration method. Their main use is in the source selection stages, in `txpipe/source_selector.py`. Their output is saved into the :ref:`tomography file<Shear tomography catalogs>` "response" group.

The calculator classes are designed to operate on streamed data (i.e. they do not load the entire catalog at once, but a chunk at a time).  The expected life cycle of a calculator class is:

1. Instantiate the calculator class with a `selector` function (see below) and other options.
2. Call the `add_data` method repeatedly with chunks of data (can be done in parallel).
3. Call the `collect` method at the end to get the final calibration factors.

The **selector** function is used because in some cases we need to calculate a selection bias quantity, and to do this we run the selection on different data variants. It should take this form::

    def selector(data: dict[str,np.ndarray], *args, **kwargs):
        ...
        return selection

The `data` argument is a dictionary of numpy arrays, where the keys are the column names and the values are the data. The function should return something we can use to index the data, such as a boolean array or an integer array.  The extra arguments are passed in from the `add_data` method, and can be anything used in the selection, such the index of the tomographic bin being selected. They will be passed in when calling `add_data`.

The **add_data** method should be called with each chunk of data loaded in one-by-one. Its `data` argument is the same as the `selector` function, and the `args` and `kwargs` are passed to the `selector` function. The 

Calibrator Classes
------------------

The calibrator classes read the data from the tomography file response group and apply it to shear catalogs, maps, or similar values. Theu can be found in `txpipe/utils/calibrators.py`. The primary place they are used is to generate calibrated tomographic catalogs in the `TXShearCalibration` stage in` `calibrate.py`

The base class, `Calibrator`, has a `load` method which takes the name of the tomography file to load from, and then determines which shear catalog type it is and returns a `Calibrator` sub-class appropriate for that type. This method also takes a "null" keyword which you can use to tell it to ignore the calibration factors and return a dummy calibrator that does nothing. This is useful when you have true shear values for testing.

All the subclasses then have an `apply` method which takes `g1` and `g2` arrays/floats and returns calibrated values. By default a mean shear value is also subtracted.

MeanShearInBins
---------------



Roadmap
-------
It would make more sense for the calculator classes to return a Calibrator object, and for the calibrator objects to be responsible for saving themselves to the tomography files where they end up.