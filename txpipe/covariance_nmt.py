# This module previously contained TXFourierNamasterCovariance and
# TXRealNamasterCovariance. These classes have been removed and replaced by
# the unified TXFourierCovariance and TXRealCovariance in covariance.py,
# which use TJPCov v0.5 via its CovarianceCalculator interface. The choice
# of covariance type (including NaMaster-based methods) is now specified
# through the ``cov_type`` configuration option.
