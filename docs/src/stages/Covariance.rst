Covariance
==========

These stages compute covariances of measurements

* :py:class:`~txpipe.covariance.TXFourierGaussianCovariance` - Compute a Gaussian Fourier-space covariance with TJPCov using f_sky only

* :py:class:`~txpipe.covariance.TXRealGaussianCovariance` - Compute a Gaussian real-space covariance with TJPCov using f_sky only

* :py:class:`~txpipe.covariance.TXFourierTJPCovariance` - Compute a Gaussian Fourier-space covariance with TJPCov using mask geometry

* :py:class:`~txpipe.covariance_nmt.TXFourierNamasterCovariance` - Compute a Gaussian Fourier-space covariance with NaMaster

* :py:class:`~txpipe.covariance_nmt.TXRealNamasterCovariance` - Compute a Gaussian real-space covariance with NaMaster



.. autoclass:: txpipe.covariance.TXFourierGaussianCovariance

    **parallel**: No - Serial

.. autoclass:: txpipe.covariance.TXRealGaussianCovariance

    **parallel**: No - Serial

.. autoclass:: txpipe.covariance.TXFourierTJPCovariance

    **parallel**: Yes - MPI

.. autoclass:: txpipe.covariance_nmt.TXFourierNamasterCovariance

    **parallel**: Yes - MPI

.. autoclass:: txpipe.covariance_nmt.TXRealNamasterCovariance

    **parallel**: Yes - MPI
