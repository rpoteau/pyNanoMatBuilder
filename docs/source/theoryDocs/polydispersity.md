# Polydispersity

## `NanoparticleDistribution` class

### Context

The `NanoparticleDistribution` class is designed to provide a comprehensive statistical analysis of nanoparticle samples. In a typical workflow, your input data, `sizes` and `counts`) originates from the analysis of raw characterization data, such as **HRTEM images**.

By performing a non-linear least squares Gaussian fit, it allows researchers to determine the mean particle size, the **Standard Deviation ($\sigma$)**, and the **Polydispersity** (Coefficient of Variation).

This module goes beyond simple curve fitting by providing:
* **Theoretical Population Mapping**: Automatic calculation of how many particles fall within specific size ranges ($1\sigma, 2\sigma, 3\sigma$).
* **Custom Binning**: The ability to discretize the theoretical curve into specific size "bins" for direct comparison with experimental histograms.
* **Publication-Ready Visualization**: Integrated plotting with statistical markers for **FWHM** (Full Width at Half Maximum) and the distribution spread.

### Input data

To perform an accurate analysis, the `NanoparticleDistribution` class expects two synchronized arrays (or lists). These represent the size distribution of your measurements (typically obtained via TEM or SAXS fitting):

1. **Count Data (`counts`)** = number of nanoparticles (or the normalized intensity) observed for each corresponding size in the `sizes` array (format: integers)

2. **Size Data (`sizes`)** = independent variable representing the different particle diameters measured on the TEM image
   * **Format**: A 1D array-like (list, numpy array, or pandas series) of floats.
   * **Unit**: nanometers (nm).

> **Note**: Both arrays must have the **same length**. The accuracy of the Gaussian fit depends on having a sufficient number of data points around the peak of the distribution.

> **Requirement**: The `sizes` values should represent the **center** of each bin in your experimental histogram.




