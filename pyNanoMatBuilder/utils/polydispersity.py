############################################################
#                    polydispersity
############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm
import os, io
from pathlib import Path

from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color

class NanoparticleDistribution:
    """
    A class to analyze and fit nanoparticle size distributions using Gaussian models.
    
    This tool provides methods for curve fitting, statistical breakdown by size bins, 
    and publication-ready visualization.
    """
    bin_width_nm = None

    def __init__(self):
        """
        Initialize the distribution.
        """        
        self.params = None  # Will store [A, mu, sigma] after fitting
        self.cov = None     # Covariance matrix for error analysis
        self.model_type = 'gaussian'  # Default model
        self._results_dict = {}       # Private storage for results
        self.bin_width_nm = None

    @classmethod
    def from_TEM(cls, sizes, counts):
        """
        Args:
            sizes (array-like): Measured particle sizes (e.g., in nm).
            counts (array-like): Number of nanoparticles for each size.
        """
        instance=cls()
        instance.sizes = np.array(sizes)
        instance.counts = np.array(counts)
        instance.model_type = 'gaussian'
        instance.bin_width_nm = None
        return instance

    @classmethod
    def from_gaussian_params(cls, mu, sigma, total_n=1000):
        """
        Instantiate the class using specific Gaussian parameters and a total population.

        This factory method calculates the peak amplitude (A) required to ensure 
        the integral of the distribution equals the target population (total_n). 
        It also generates initial binned pseudo-data (sizes and counts) centered 
        around the mean.

        Args:
            mu (float): Mean diameter of the particles (nm).
            sigma (float): Standard deviation of the distribution (nm).
            total_n (int/float, optional): Total number of nanoparticles 
                in the distribution. Defaults to 1000.

        Returns:
            NanoparticleDistribution: An instance initialized with a theoretical 
                Gaussian model and generated binned data.

        Note:
            The bin width (w) defaults to the class-level bin_width_nm if set; 
            otherwise, it defaults to sigma. The generated range covers mu ± 4σ.
        """
        instance = cls()
        
        # Calculate amplitude to ensure the sum (area) equals total_n
        # A = N / (sigma * sqrt(2 * pi))
        amplitude = total_n / (sigma * np.sqrt(2 * np.pi))
        
        instance.params = np.array([amplitude, mu, sigma])
        
        # Generate enough points so that np.sum(counts * dx) is accurate
        # But for your binned stats, we just need to store the intention
        w = cls.bin_width_nm if cls.bin_width_nm is not None else sigma
        instance.bin_width_nm = w
        # Create symmetrical bin centers around mu
        # We go up to ~4 sigma to cover the distribution
        half_range = 4 * sigma
        num_steps = int(half_range / w)
        
        # This generates: [mu - N*w, ..., mu, ..., mu + N*w]
        instance.sizes = mu + np.arange(-num_steps, num_steps + 1) * w
        # instance.sizes = np.linspace(mu - 4*sigma, mu + 4*sigma, 10)
        instance.counts = instance._gaussian_model(instance.sizes, *instance.params)
        
        # We manually set a 'total_n' attribute or just rely on the math
        instance.total_n_expected = total_n 
        
        instance.cov = np.zeros((3, 3))
        instance.total_n_expected = total_n
        instance.model_type = 'gaussian'
        return instance

    @classmethod
    def from_schulz_params(cls, mu, pd_pct, total_n=1000):
        """
        Instantiate the class using the Schulz distribution.
    
        The Schulz distribution is skewed towards larger values, making it
        well-suited for nanoparticle size distributions from SAXS/SANS experiments.
    
        Args:
            mu (float): Mean diameter of the particles (nm).
            pd_pct (float): Polydispersity percentage (sigma/mu * 100).
            total_n (int/float, optional): Total number of particles. Defaults to 1000.
    
        Returns:
            NanoparticleDistribution: An instance with model_type='schulz'.
        """
        instance = cls()
        p = pd_pct / 100.0
        z = (1.0 - p**2) / p**2      # Schulz width parameter
        sigma = p * mu                # RMS deviation
    
        # params stores (mu, z) — no separate amplitude, total_n drives scaling
        instance.params = np.array([mu, z])
        instance.model_type = 'schulz'
        instance.total_n_expected = total_n
    
        # Generate initial bin centers for display (±4σ, clipped at 0)
        w = cls.bin_width_nm if cls.bin_width_nm is not None else sigma
        instance.bin_width_nm = w
        half_range = 4.0 * sigma
        num_steps = int(half_range / w)
        instance.sizes = np.clip(
            mu + np.arange(-num_steps, num_steps + 1) * w,
            1e-6, None
        )
        instance.counts = (
            instance._schulz_model(instance.sizes, mu, z) * total_n * w
        )
    
        instance.cov = np.zeros((2, 2))
        return instance
    
    @classmethod
    def from_polydispersity(cls, mu, pd_pct, amplitude=1000, model='gaussian'):
        """
        Instantiate the class using mean diameter and Polydispersity Index (CV%).

        This method acts as a convenience wrapper for from_gaussian_params, 
        automatically calculating the standard deviation (sigma) based on the 
        Coefficient of Variation (CV% or PD%).

        Formula:
            sigma = (PD% / 100) * mu

        Args:
            mu (float): Mean diameter of the particles (nm).
            pd_pct (float): Polydispersity percentage (Coefficient of Variation %).
            amplitude (int/float, optional): Total number of nanoparticles 
                to be simulated (Total N). Defaults to 1000.
            model (str): 'gaussian' or 'schulz'. Defaults to 'gaussian'.

        Returns:
            NanoparticleDistribution: An instance initialized with the derived 
                Gaussian parameters.
        """
        sigma = (pd_pct / 100) * mu
        
        if model == 'gaussian':
            return cls.from_gaussian_params(mu, sigma, amplitude)
    
        elif model == 'schulz':
            return cls.from_schulz_params(mu, pd_pct, amplitude)
    
        else:
            raise ValueError(f"Unknown model '{model}'. Choose 'gaussian' or 'schulz'.")

    @staticmethod
    def _gaussian_model(x, A, mu, sigma):
        """
        Internal Gaussian model function.
        
        Args:
            x (float/array): Size values.
            A (float): Amplitude (Peak height).
            mu (float): Mean (Center of distribution).
            sigma (float): Standard deviation (Width).
            
        Returns:
            float/array: Probability density values.
        """
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

    @staticmethod
    def _schulz_model(x, mu, z):
        """
        Normalized Schulz probability density function.
    
        The distribution is parameterized by its mean (mu) and width
        parameter z = (1 - p^2) / p^2, where p = sigma / mu.
    
        Args:
            x (float or np.ndarray): Size values (must be > 0).
            mu (float): Mean of the distribution.
            z (float): Width parameter (large z → narrow distribution).
    
        Returns:
            np.ndarray: Probability density values (normalized, area = 1).
        """
        from scipy.special import gamma
        x = np.asarray(x, dtype=float)
        coeff = (z + 1)**(z + 1) / (mu * gamma(z + 1))
        return coeff * (x / mu)**z * np.exp(-(z + 1) * x / mu)
    
    @staticmethod
    def _schulz_cdf_bin(s1, s2, mu, z):
        """
        Probability mass of the Schulz distribution over the interval [s1, s2].
    
        Uses the regularized incomplete gamma function (scipy.special.gammainc)
        for numerical accuracy on any bin width.
    
        Args:
            s1 (float): Lower bin edge (nm).
            s2 (float): Upper bin edge (nm).
            mu (float): Mean of the distribution.
            z (float): Width parameter.
    
        Returns:
            float: Probability P(s1 ≤ X < s2).
        """
        from scipy.special import gammainc
        # CDF(x) = gammainc(z+1, (z+1)*x/mu)  [regularized lower incomplete gamma]
        a = z + 1.0
        cdf = lambda x: gammainc(a, a * x / mu)   # noqa: E731
        return float(np.clip(cdf(s2) - cdf(s1), 0, 1))

    def _compute_fwhm(self, mu, sigma):
        """
        Compute the Full Width at Half Maximum (FWHM) of the distribution.
    
        For Gaussian, the result is exact and analytical.
        For Schulz, the distribution is asymmetric so the half-maximum
        points are found numerically via Brent's method, using the mode
        (x_mode = z/(z+1) * mu) as the pivot between left and right searches.
    
        Args:
            mu (float): Mean of the distribution (nm).
            sigma (float): Standard deviation (nm). Used directly for Gaussian;
                for Schulz, z is retrieved from self.params instead.
    
        Returns:
            tuple: (fwhm, x_left, x_right) where:
                - fwhm (float): Full width at half maximum (nm).
                - x_left (float): Left half-maximum position (nm).
                - x_right (float): Right half-maximum position (nm).
    
        Raises:
            NotImplementedError: If model_type is not supported.
        """
        if self.model_type == 'gaussian':
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            return fwhm, mu - fwhm / 2, mu + fwhm / 2
    
        elif self.model_type == 'schulz':
            from scipy.optimize import brentq
            z = self.params[1]
            x_mode   = z / (z + 1) * mu
            half_max = self._schulz_model(x_mode, mu, z) / 2
            x_left   = brentq(lambda x: self._schulz_model(x, mu, z) - half_max, 1e-6, x_mode)
            x_right  = brentq(lambda x: self._schulz_model(x, mu, z) - half_max, x_mode, mu * 10)
            return x_right - x_left, x_left, x_right
    
        else:
            raise NotImplementedError(f"_compute_fwhm() not implemented for model_type='{self.model_type}'.")
        
    def fit(self, p0=None):
        """
        Perform a non-linear least squares Gaussian fit on the data.
        
        Args:
            p0 (list, optional): Initial guesses for [A, mu, sigma]. 
                                 Defaults to automatic estimates from data.
                                 
        Returns:
            numpy.ndarray: Optimized parameters [A, mu, sigma].
        """
        if p0 is None:
            # Automatic guess: [Peak height, average size, rough spread]
            p0 = [np.max(self.counts), np.mean(self.sizes), 0.2]
        
        self.params, self.cov = curve_fit(self._gaussian_model, self.sizes, self.counts, p0=p0)
        self.print_accuracy_report()
        return self.params

    def get_fit_accuracy(self):
        """
        Calculate the statistical uncertainty of the fit parameters and the R-squared value.

        This method extracts the standard errors (1-sigma uncertainty) for the 
        distribution parameters from the diagonal of the covariance matrix. 
        It also calculates the R-squared (coefficient of determination) to 
        quantify how well the Gaussian model accounts for the variance in the data.

        Returns:
            dict: A dictionary containing:
                - "mu_error": Standard error of the mean (nm).
                - "sigma_error": Standard error of the standard deviation (nm).
                - "r_squared": The R-squared value (0 to 1).

        Raises:
            ValueError: If the covariance matrix (self.cov) is None, indicating 
                that the fit() method has not been successfully executed.
        """
        if self.cov is None:
            raise ValueError("Fit the model first.")

        # 1. Parameter Errors (1-sigma uncertainty)
        perr = np.sqrt(np.diag(self.cov))
        
        # 2. R-squared calculation
        residuals = self.counts - self._gaussian_model(self.sizes, *self.params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.counts - np.mean(self.counts))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            "mu_error": perr[1],
            "sigma_error": perr[2],
            "r_squared": r_squared
        }

    def print_accuracy_report(self):
        """
        Print a formatted statistical report on the reliability of the Gaussian fit.

        This method retrieves the R-squared value and parameter uncertainties 
        from get_fit_accuracy() and displays them alongside a qualitative 
        assessment (Reliable/Acceptable/Poor) based on the fit quality.

        Args:
            None: This method uses the results and errors stored in the instance.

        Returns:
            None: Outputs a formatted summary to the console.

        Note:
            This method indirectly requires self.cov to be defined. If fit() 
            has not been called, the internal call to get_fit_accuracy() 
            will raise a ValueError.
        """
        acc = self.get_fit_accuracy()
        res = self.results
        
        centerTitle('Fit Accuracy Report')
        print(f"R-squared (Fit quality) : {acc['r_squared']:.4f}")
        print(f"Mean Size Accuracy      : {res['mean']:.3f} ± {acc['mu_error']:.3f} nm")
        print(f"Sigma Accuracy         : {res['sigma']:.3f} ± {acc['sigma_error']:.3f} nm")
        
        if acc['r_squared'] > 0.95:
            print(f"{fg.GREEN}Highly reliable fit.{fg.OFF}")
        elif acc['r_squared'] > 0.85:
            print(f"{fg.ORANGE}Acceptable fit.{fg.OFF}")
        else:
            print(f"{fg.RED}Poor fit. Consider checking for outliers or asymmetry.{fg.OFF}")

    @property
    def results(self):
        """
        Calculate physical parameters from the fitted model.
        
        Returns:
            dict: Keys depend on model_type:
                - Gaussian: amplitude, mean, sigma, fwhm, cv_percentage
                - Schulz:   amplitude (=total_n), mean, sigma, z, fwhm, cv_percentage
    
        Raises:
            ValueError: If called before parameters are set.
        """

        if self.params is None:
            raise ValueError("Fit has not been performed yet.")
        
        # Handle Gaussian logic
        if self.model_type == 'gaussian':
            A, mu, sigma = self.params
            fwhm, _, _  = self._compute_fwhm(mu, sigma)
            cv = (sigma / mu) * 100
            return {
                "amplitude": A,
                "mean": mu,
                "sigma": sigma,
                "fwhm": fwhm,
                "cv_percentage": cv
            }
        elif self.model_type == 'schulz':
            mu, z   = self.params
            p       = 1.0 / np.sqrt(z + 1.0)
            sigma   = p * mu
            fwhm, _, _ = self._compute_fwhm(mu, sigma)
            cv = p * 100.0
            amplitude = getattr(self, 'total_n_expected', 1000)
            return {
                "amplitude": amplitude,
                "mean": mu,
                "sigma": sigma,
                "z": z,
                "fwhm": fwhm,
                "cv_percentage": cv,
            }
        else:
            raise NotImplementedError(f"results() not implemented for model_type='{self.model_type}'.")
    
    def print_results(self):
        """
        Display a formatted summary of the distribution's statistical properties.

        This method outputs basic parameters (Mean, Amplitude, Polydispersity) 
        and calculates the theoretical population coverage for both Gaussian 
        and Log-normal models. It dynamically computes the probability of 
        finding a particle within 1, 2, and 3 standard deviations (or geometric 
        steps) from the center.

        Args:
            None: Retrieves data from the results property and internal state.

        Returns:
            None: Prints a formatted summary directly to the console.

        Note:
            For Log-normal models, the coverage is calculated using geometric 
            multiples of the geometric standard deviation (sigma_g) around 
            the median.
        """
        from scipy.stats import norm
        
        centerTitle("Summary of the distribution statistics")
        stats = self.results
        model = getattr(self, 'model_type', 'gaussian')

        # Basic Stats
        print(f"Model Type          : {model.upper()}")
        print(f"Amplitude           : {stats['amplitude']:.0f} particles")
        print(f"Average Size (Mean) : {stats['mean']:.3f} nm")
        print(f"Polydispersity (CV) : {stats['cv_percentage']:.2f}%")

        if model == 'gaussian':
            mu, sigma = stats['mean'], stats['sigma']
            
            # Dynamic calculation function for Gaussian
            def get_prob(low, high):
                return (norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)) * 100

            print(f"Std Deviation (σ)   : {sigma:.3f} nm")
            print("-" * 40)
            print(f"Theoretical Population Coverage (Dynamic):")
            print(f"  μ ± 1σ     ({mu-sigma:>5.2f}-{mu+sigma:<5.2f} nm) : {get_prob(mu-sigma, mu+sigma):.1f}%")
            print(f"  μ ± 2σ     ({mu-2*sigma:>5.2f}-{mu+2*sigma:<5.2f} nm) : {get_prob(mu-2*sigma, mu+2*sigma):.1f}%")
            print(f"  μ ± 3σ     ({mu-3*sigma:>5.2f}-{mu+3*sigma:<5.2f} nm) : {get_prob(mu-3*sigma, mu+3*sigma):.1f}%")

    def get_relative_height(self, x_value):
        """
        Calculate height relative to the distribution peak.
    
        For Gaussian: uses the standard Gaussian ratio exp(-0.5 * z^2).
        For Schulz:   evaluates f(x) / f(mu) analytically.
    
        Args:
            x_value (float or np.ndarray): Size value(s) in nm.
    
        Returns:
            float or np.ndarray: Relative height in [0, 1].
        """
        mu = self.results['mean']
        sigma = self.results['sigma']

        if self.model_type == 'gaussian':
            z_score = (x_value - mu) / sigma
            return np.exp(-0.5 * z_score**2)

        elif self.model_type == 'schulz':
            z = self.results['z']
            x = np.asarray(x_value, dtype=float)
            ratio = x / mu
            return ratio**z * np.exp(-(z + 1.0) * (ratio - 1.0))
    
        else:
            raise NotImplementedError(f"get_relative_height() not implemented for model_type='{self.model_type}'.")
        
        
    def get_binned_statistics(self, bin_width_nm=None, total_n=None):
        """
        Calculate and display theoretical populations using fixed-width size bins.

        This method discretizes the continuous Gaussian or Log-normal distribution into 
        bins of a specified width. It calculates the expected particle count for each 
        bin using the Cumulative Distribution Function (CDF) and determines the 
        relative height of each bin center compared to the peak.

        The results are printed in a formatted table and the instance's plotting 
        data (self.sizes, self.counts) is synchronized to reflect the new binning.

        Args:
            bin_width_nm (float, optional): The width of each size bin in nanometers. 
                If not provided, it follows a priority chain: 
                Instance setting > Class setting > Sigma (fallback).
            total_n (int/float, optional): The total number of particles to distribute. 
                Defaults to the value stored during instantiation or 1000.

        Returns:
            None: Outputs a summary table to the console and updates internal state.

        Note:
            - For Gaussian models, bins are generated symmetrically around the mean.
            - Synchronization: Updating the bins here will immediately change the 
              appearance of the histogram in the plot() method.
        """
        from scipy.stats import norm

        if self.params is None and not hasattr(self, '_results_dict'):
            raise ValueError(f"No parameters available. Fit the model or instantiate from params first.")

        # Extract stats from the unified results property
        stats = self.results
        mu = stats['mean']
        sigma = stats['sigma']
        
        if bin_width_nm is not None:
            w = bin_width_nm
            self.bin_width_nm = bin_width_nm # On ne touche PAS à NanoparticleDistribution.bin_width_nm
        elif self.bin_width_nm is not None:
            w = self.bin_width_nm
        elif self.__class__.bin_width_nm is not None:
            w = self.__class__.bin_width_nm
        else:
            w = sigma
        
        # 2. Logic to pick the best available width
        # Priority: Instance > Class > Sigma
        w = self.bin_width_nm if self.bin_width_nm else (self.__class__.bin_width_nm if self.__class__.bin_width_nm else sigma)
        
        # Update the local helper 'w' and ensure the instance knows it
        self.bin_width_nm = w

        if total_n is None:
            # 1. On regarde d'abord si une valeur a été stockée explicitement
            if hasattr(self, 'total_n_expected') and self.total_n_expected is not None:
                total_n = self.total_n_expected
            # 2. En dernier recours, on somme (cas des vraies données expérimentales)
            elif len(self.counts) > 0 and self.cov is not None and np.any(self.cov > 0):
                total_n = np.sum(self.counts)
            else:
                total_n = 1000
            
        # 1. Define coverage limits (±3.5 sigma)
        # For lognormal, we must ensure we don't go below or equal to zero
        limit_min = mu - 3.5 * sigma
        limit_max = mu + 3.5 * sigma
        
        # 2. Generate bin edges starting from mu to ensure symmetry (for Gaussian)
        # or just a consistent range for Log-normal
        right_edges = np.arange(mu + w/2, limit_max + w, w)
        left_edges = np.arange(mu - w/2, limit_min - w, -w)
        edges = np.sort(np.concatenate([left_edges, right_edges]))
        if self.model_type == 'schulz':
            edges = edges[edges > 0]   # Schulz is defined on (0, +inf) only
        
        # --- 1. First pass: calculate data and sum of ratios for normalization ---
        bins_results = []
        total_ratio_sum = 0
        
        for i in range(len(edges) - 1):
            s1, s2 = edges[i], edges[i+1]
            bin_center = (s1 + s2) / 2
            
            if self.model_type == 'schulz':
                prob = self._schulz_cdf_bin(max(s1, 1e-6), s2, mu, self.results['z'])
            elif self.model_type == 'gaussian':
                prob = norm.cdf(s2, mu, sigma) - norm.cdf(s1, mu, sigma)
            ratio_to_peak = self.get_relative_height(bin_center)
            
            total_ratio_sum += ratio_to_peak
            bins_results.append({
                'range': f"[{s1:>5.2f}, {s2:<5.2f}[",
                'center': (s2 + s1)/2,
                'count': prob * total_n,
                'prob': prob,
                'ratio': ratio_to_peak
            })

        # --- 2. Formatting and Output ---
        w_range, w_count, w_prob, w_ratio, w_norm = 18, 10, 10, 12, 12
        
        centerTitle(f'Binned Population (Step={w:.3f} nm, N={total_n})')
        
        header = (f"{'Size Range (nm)':<{w_range}} | {'Count':<{w_count}}| "
                  f"{'Area (%)':<{w_prob}}| {'Ratio/Peak':<{w_ratio}}| {'Norm. (1)':<{w_norm}}")
        print(header)
        print("-" * len(header))
        
        running_count, running_prob, running_norm = 0, 0, 0
        
        new_sizes = []
        new_counts = []
        for b in bins_results:
            # Normalize so that the sum of the column equals 1.000
            norm_val = b['ratio'] / total_ratio_sum if total_ratio_sum > 0 else 0
            
            running_count += b['count']
            running_prob += b['prob']
            running_norm += norm_val

            new_sizes.append(b['center']) 
            new_counts.append(b['count'])
            
            print(f"{b['range']:<{w_range}} | "
                  f"{int(round(b['count'])):>{w_count-1}} | "
                  f"{b['prob']*100:>{w_prob-2}.1f}% | "
                  f"{b['ratio']:>{w_ratio-2}.3f} | "
                  f"{norm_val:>{w_norm-2}.3f}")
            
        self.sizes = np.array(new_sizes)
        self.counts = np.array(new_counts)

        print("-" * len(header))
        print(f"{'Total Covered':<{w_range}} | "
              f"{int(round(running_count)):>{w_count-1}} | "
              f"{running_prob*100:>{w_prob-2}.1f}% | "
              f"{'-':>{w_ratio-2}} | "
              f"{running_norm:>{w_norm-2}.3f}")

    def get_proportions(self, target_sizes, labels=None, bin_width_nm=None):
        targets = np.atleast_1d(target_sizes)
    
        if labels is None:
            resolved_labels = [''] * len(targets)
        else:
            resolved_labels = list(labels)
    
        stats = self.results
        mu, sigma = stats['mean'], stats['sigma']
    
        # --- Bin width ---
        if bin_width_nm is None:
            bin_width_nm = self.bin_width_nm if self.bin_width_nm else sigma
    
        # --- Coverage limits ---
        limit_min = mu - 3.5 * sigma
        limit_max = mu + 3.5 * sigma
    
        # --- Bin edges ---
        r_edges = np.arange(mu + bin_width_nm / 2, limit_max + bin_width_nm, bin_width_nm)
        l_edges = np.arange(mu - bin_width_nm / 2, limit_min - bin_width_nm, -bin_width_nm)
        edges = np.sort(np.concatenate([l_edges, r_edges]))
        if self.model_type == 'schulz':
            edges = edges[edges > 0]   # Schulz defined on (0, +inf) only
    
        # --- total_ratio_sum via get_relative_height (model-aware) ---
        total_ratio_sum = 0
        for i in range(len(edges) - 1):
            bc = (edges[i] + edges[i + 1]) / 2
            total_ratio_sum += self.get_relative_height(bc)
    
        # --- Ratios for targets via get_relative_height (model-aware) ---
        ratios = np.array([self.get_relative_height(t) for t in targets])
    
        # --- Normalization ---
        norm_values  = ratios / total_ratio_sum if total_ratio_sum > 0 else np.zeros_like(ratios)
        norm_sum     = ratios.sum()
        norm_relative = ratios / norm_sum if norm_sum > 0 else np.zeros_like(ratios)
        counts = ratios * stats['amplitude']
    
        return {
            "sizes"         : targets,
            "labels"        : resolved_labels,
            "ratios"        : ratios,
            "counts"        : counts,
            "norms"         : norm_values,
            "norms_relative": norm_relative,
        }
            
    def print_specific_proportions(self, target_sizes, labels=None):
        """
        Print a formatted summary of proportions for specific diameters.
        """
        data = self.get_proportions(target_sizes, labels=labels)
        
        centerTitle("Specific Diameter Proportions")
        has_labels = any(l != '' for l in data['labels'])
        
        label_col = f"{'Label':<8} | " if has_labels else ""
        header = (f"{label_col}{'Diameter (nm)':<15} | {'Ratio/Peak':<12} | "
                  f"{'Est. Count':<12} | {'Norm. (dist)':<14} | {'Norm. (1)':<12}")
        print(header)
        print("-" * len(header))
    
        for i in range(len(data['sizes'])):
            label_str = f"{data['labels'][i]:<8} | " if has_labels else ""
            print(f"{label_str}"
                  f"{data['sizes'][i]:>12.2f} nm | "
                  f"{data['ratios'][i]:>10.3f}   | "
                  f"{data['counts'][i]:>10.0f}   | "
                  f"{data['norms'][i]:>12.4f}   | "
                  f"{data['norms_relative'][i]:>10.4f}")
        print("-" * len(header))
        
    def plot(self, title='Nanoparticle Size Distribution', color_histo="skyblue",
             color_curve="red", plot_histogram=True, highlight_sizes=None,
             save_img=None, dpi=300):
        """
        Visualize the nanoparticle size distribution (histogram) overlaid with the 
        fitted or theoretical Gaussian model.

        The method automatically detects the context: 
        - For experimental fits, it plots raw counts.
        - For theoretical simulations, it scales the Gaussian PDF by the bin width 
          to ensure the curve peak aligns visually with the histogram bars.

        Args:
            title (str): Graph title displayed at the top.
            color_histo (str): Color name or hex code for the histogram bars.
            color_curve (str): Color name or hex code for the Gaussian Or Schuz fit line.
            plot_histogram (bool): If True, superimposes the bars over the curve. 
                Requires sizes and counts to be initialized.
            highlight_sizes (list/array, optional): Specific diameters (nm) to mark 
                with dots and labels on the distribution curve.
            save_img (str, optional): File path or name (e.g., 'plot.png' or 'plot.svg'). 
                The directory is created automatically if it doesn't exist.
            dpi (int): Image resolution for raster formats like PNG (default 300).

        Returns:
            None: Displays the plot using plt.show() and optionally saves to disk.
        """
        # --- 1. Safety Check ---
        # Ensure model parameters exist before attempting to plot
        if self.params is None:
            raise ValueError("Cannot plot: Distribution parameters are not defined. "
                             "Please run fit() or instantiate using from_gaussian_params().")
            
        res = self.results
        mu = res['mean']
        sigma = res['sigma']
        plt.figure(figsize=(10, 6))
        
        # Calculate bar width based on data spacing
        bar_width = (self.sizes[1] - self.sizes[0]) * 0.9
        
        # --- 2. Scaling Logic ---
        # If the model was fitted (covariance exists), use raw amplitude (scaling = 1.0).
        # In simulation mode, scale the PDF by the bin width to match histogram counts.
        if self.cov is not None and np.any(self.cov > 0):
            scaling_w = 1.0
            histo_label = "Exp. data" # Legend for real TEM data
        else:
            scaling_w = self.bin_width_nm if self.bin_width_nm else 1.0
            histo_label = f"Binned Model (w={scaling_w:.2f} nm)" # Legend for simulation

        # --- 3. Plotting Data and Fit ---
        if plot_histogram and self.sizes is not None: plt.bar(self.sizes, self.counts, width=bar_width, color=color_histo, label=histo_label)    
            
        # --- 4. Distribution Curve ---
        # Generate smooth x values for the curve
        x_smooth = np.linspace(self.sizes.min() * 0.8, self.sizes.max() * 1.2, 500)
        
        # Apply scaling to the model for visual alignment
        if self.model_type == 'gaussian':
            y_smooth = self._gaussian_model(x_smooth, *self.params) * scaling_w
            model_name = "Gaussian"
        
        elif self.model_type == 'schulz':
            mu, z = self.params
            # Scale PDF so its peak aligns with histogram bars
            y_smooth = self._schulz_model(x_smooth, mu, z) * scaling_w * self.results['amplitude']
            model_name = "Schulz"
        
        extra = f"\n$z$ = {self.results['z']:.1f}" if self.model_type == 'schulz' else ""
        label_text = (f"{model_name} {'Fit' if scaling_w == 1.0 else 'Model'}:\n"
                      f"$\mu$ = {res['mean']:.2f} nm\n"
                      f"$\sigma$ = {res['sigma']:.2f} nm\n"
                      f"Polydispersity = {res['cv_percentage']:.1f}%"
                      f"{extra}")
        
        plt.plot(x_smooth, y_smooth, color=color_curve, lw=2, label=label_text)

        # Plot FWHM (Full Width at Half Maximum) indicator line
        fwhm, x_left, x_right = self._compute_fwhm(mu, sigma)
        # print(f"model={self.model_type}, mu={mu}, sigma={sigma}, fwhm={fwhm}")
        if self.model_type == 'gaussian':
            y_half = res['amplitude'] / 2 * scaling_w
        elif self.model_type == 'schulz':
            x_mode = self.params[1] / (self.params[1] + 1) * mu
            y_half = self._schulz_model(x_mode, *self.params) / 2 * scaling_w * res['amplitude']
        plt.hlines(y=y_half, xmin=x_left, xmax=x_right,
                   colors='green', linestyles='--',
                   label=f"FWHM ({fwhm:.2f} nm)")

        # --- 5. Statistical Overlays ---
        # Vertical lines at +/- 1 and 2 sigma to visualize spread
        plt.axvline(x=mu - sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 1\sigma$ (Spread)")
        plt.axvline(x=mu + sigma, color='#3f8188', linestyle=':', lw=1.5)
        plt.axvline(x=mu - 2*sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 2\sigma$ (Spread)")
        plt.axvline(x=mu + 2*sigma, color='#3f8188', linestyle=':', lw=1.5)
        plt.axvline(x=mu - 3*sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 3\sigma$ (Spread)")
        plt.axvline(x=mu + 3*sigma, color='#3f8188', linestyle=':', lw=1.5)
        plt.axvline(x=mu - 4*sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 4\sigma$ (Spread)")
        plt.axvline(x=mu + 4*sigma, color='#3f8188', linestyle=':', lw=1.5)

        # --- 6. Specific Point Highlighting ---
        # Parse highlight_sizes: either a flat list or [sizes, labels]
        h_sizes, h_labels = None, None
        if highlight_sizes is not None:
            if (isinstance(highlight_sizes, (list, tuple)) and len(highlight_sizes) == 2
                    and isinstance(highlight_sizes[0], (list, np.ndarray))
                    and isinstance(highlight_sizes[1], (list, np.ndarray))):
                h_sizes = list(highlight_sizes[0])
                h_labels = list(highlight_sizes[1])
            else:
                h_sizes = list(highlight_sizes)
                h_labels = None
    
        if h_sizes is not None:
            props = self.get_proportions(h_sizes)
            
            for i, size in enumerate(h_sizes):
                current_norm = props['norms_relative'][i]
                if self.model_type == 'gaussian':
                    y_pos = self._gaussian_model(size, *self.params) * scaling_w
                elif self.model_type == 'schulz':
                    y_pos = self._schulz_model(size, *self.params) * scaling_w * res['amplitude']
                
                # Build annotation text
                label_prefix = f"{h_labels[i]}: " if h_labels is not None else ""
                label_line = f"{h_labels[i]}\n" if h_labels is not None else ""
                annotation = f"{label_line}{size:.2f}nm\n({current_norm:.3f})"
                
                plt.scatter(size, y_pos, color='black', zorder=5)
                plt.annotate(annotation,
                             (size, y_pos), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
        # --- 7. Final Formatting ---
        plt.title(title)
        plt.xlabel('Size (nm)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        # Add headroom if highlight annotations are present
        if highlight_sizes is not None:
            y_min, y_max = plt.ylim()
            plt.ylim(y_min, y_max * 1.1)
        plt.tight_layout()
                        
        # --- 8. Enhanced Save Logic ---
        if save_img:
            save_path = Path(save_img)
            
            # Ensure the directory exists
            if save_path.parent:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            ext = save_path.suffix.lower()
            
            if ext == '.svg':
                plt.savefig(save_path, format='svg')
                print(f"Plot saved as vector graphics: {save_path}")
            elif ext == '.png':
                plt.savefig(save_path, format='png', dpi=dpi)
                print(f"Plot saved as PNG: {save_path}")
            else:
                # Default fallback for other formats supported by matplotlib (pdf, jpg, etc.)
                plt.savefig(save_path)
                print(f"Plot saved with {ext} format: {save_path}")

        plt.show()

    @property
    def representative_sizes(self):
        """
        Returns the representative sizes at μ - 4σ, μ - 3σ, ..., μ, ..., μ + 4σ.
        Negative or zero sizes are excluded (relevant for small NPs or wide distributions).
        """
        res = self.results
        mu, sigma = res['mean'], res['sigma']
        sizes = np.array([mu + k * sigma for k in range(-4, 5)])
        return sizes[sizes > 0]
    
    @property
    def representative_labels(self):
        """
        Returns the corresponding labels: μ - 4σ, ..., μ, ..., μ + 4σ.
        """
        res = self.results
        mu, sigma = res['mean'], res['sigma']
        all_labels = [
            'μ-4σ', 'μ-3σ', 'μ-2σ', 'μ-σ', 'μ',
            'μ+σ', 'μ+2σ', 'μ+3σ', 'μ+4σ'
        ]
        sizes = np.array([mu + k * sigma for k in range(-4, 5)])
        # Keep only positive sizes
        valid = sizes > 0
        return [l for l, v in zip(all_labels, valid) if v]

    def filter_proportions(self, data, threshold=0.01):
        """
        Filter proportions above a threshold, renormalize, and sort by size.
        
        Args:
            data (dict): Output of get_proportions().
            threshold (float): Minimum normalized proportion to keep.
                              Default is 0.01 (1%).
        Returns:
            dict: Filtered, renormalized and size-sorted data,
                  same structure as get_proportions().
        """
        import numpy as np
        
        # Filter by norms_relative threshold
        mask = data['norms_relative'] >= threshold
        
        if not np.any(mask):
            print(f"Warning: no sizes above threshold {threshold}. Returning empty.")
            return data
        
        filtered_sizes          = data['sizes'][mask]
        filtered_labels         = [l for l, m in zip(data['labels'], mask) if m]
        filtered_ratios         = data['ratios'][mask]
        filtered_counts         = data['counts'][mask]
        filtered_norms          = data['norms'][mask]
        filtered_norms_relative = filtered_ratios / filtered_ratios.sum()
    
        # Sort by size (ascending)
        sort_idx = np.argsort(filtered_sizes)
        
        return {
            'sizes':          filtered_sizes[sort_idx],
            'labels':         [filtered_labels[i] for i in sort_idx],
            'ratios':         filtered_ratios[sort_idx],
            'counts':         filtered_counts[sort_idx],
            'norms':          filtered_norms[sort_idx],
            'norms_relative': filtered_norms_relative[sort_idx],
        }