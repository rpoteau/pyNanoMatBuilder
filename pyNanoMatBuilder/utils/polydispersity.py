############################################################
#                    polydispersity
############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm, lognorm
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
    def from_polydispersity(cls, mu, pd_pct, amplitude=1000):
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

        Returns:
            NanoparticleDistribution: An instance initialized with the derived 
                Gaussian parameters.
        """
        sigma = (pd_pct / 100) * mu
        return cls.from_gaussian_params(mu, sigma, amplitude)

    @classmethod
    def from_saxs_data(cls, mu_vol, pd_vol_pct, amplitude=1000):
        """
        Instantiate the class by converting SAXS (volume-weighted) parameters 
        into Number-weighted parameters using numerical cube-weighting.
        
        This method uses a robust iterative solver to find the number-weighted 
        mean (mu_n) that corresponds to the observed SAXS volume-weighted mean.
        """
        
        cv = pd_vol_pct / 100

        # --- 1. Hatch-Choate Approximation (for comparison) ---
        mu_n_hc = mu_vol / (1 + 3 * (cv**2))
        sigma_n_hc = mu_n_hc * cv

        # --- 2. Numerical Integration Approach ---
        def objective(mu_n_guess):
            # fsolve can pass an array, we need the scalar value
            m = float(mu_n_guess[0]) if isinstance(mu_n_guess, np.ndarray) else float(mu_n_guess)
            s = m * cv
            
            # Use a fixed number of points and a range based on the target mu_vol 
            # to keep the array size and limits stable for fsolve
            x = np.linspace(mu_vol * 0.1, mu_vol * 2.0, 1000)
            
            # Calculate Gaussian
            y_num = np.exp(-0.5 * ((x - m) / s)**2)
            y_vol = y_num * (x**3)
            
            # Avoid division by zero
            denom = np.trapezoid(y_vol, x)
            if denom == 0:
                return 1e6 # Penalty for invalid mu_n
                
            calc_mu_vol = np.trapezoid(y_vol * x, x) / denom
            return calc_mu_vol - mu_vol

        # Solve for the true mu_n
        # We use mu_n_hc as a much better starting guess than mu_vol
        solution = fsolve(objective, x0=mu_n_hc)
        mu_n_num = float(solution[0])
        sigma_n_num = mu_n_num * cv
        
        # --- Output Comparison ---
        print(f"\n{' SAXS to Number Conversion ':-^60}")
        print(f"Input (SAXS)       : μ={mu_vol:.3f} nm, PD={pd_vol_pct:.1f}%")
        print("-" * 60)
        print(f"{'Method':<20} | {'Mean (nm)':<15} | {'Sigma (nm)':<15}")
        print(f"{'Hatch-Choate':<20} | {mu_n_hc:>14.3f} | {sigma_n_hc:>14.3f}")
        print(f"{'Numerical (Full)':<20} | {mu_n_num:>14.3f} | {sigma_n_num:>14.3f}")
        print("-" * 60)
        
        return cls.from_gaussian_params(mu_n_num, sigma_n_num, amplitude)

    @classmethod
    def from_lognormal_params(cls, median, sigma_g, amplitude=1000):
        """
        Instantiates the class using Log-Normal parameters.
        
        Args:
            median (float): The median diameter (nm).
            sigma_g (float): The geometric standard deviation.
            amplitude (float): Total number of particles.
        """
        instance = cls()
        instance.model_type = 'lognormal'
        
        # Calculate arithmetic equivalents for summary reporting
        ln_sig_g = np.log(sigma_g)
        arith_mean = median * np.exp((ln_sig_g**2) / 2)
        arith_sigma = arith_mean * np.sqrt(np.exp(ln_sig_g**2) - 1)
        
        instance._results_dict = {
            'mean': arith_mean,
            'median': median,
            'sigma': arith_sigma,
            'sigma_g': sigma_g,
            'amplitude': amplitude,
            'cv_percentage': (arith_sigma / arith_mean) * 100,
            'fwhm': 0 # FWHM is less standard for lognormal, can be calculated if needed
        }
        
        # Generate representative data for plotting
        x = np.linspace(median / (sigma_g**2), median * (sigma_g**2), 500)
        instance.sizes = x
        instance.counts = instance._lognormal_model(x, amplitude, median, sigma_g)
        instance.params = [amplitude, median, sigma_g]
        instance.total_n_expected = amplitude
        return instance

    @classmethod
    def from_saxs_lognormal(cls, mu_vol, pd_vol_pct, total_n=1000):
        """
        Converts SAXS Volume-weighted Mean to Number-weighted Log-normal parameters
        using exact Hatch-Choate identities.
        """
        import math
        cv = pd_vol_pct / 100
        ln_sig_g_sq = math.log(1 + cv**2)
        sigma_g = math.exp(math.sqrt(ln_sig_g_sq))
        
        # Exact Hatch-Choate: median_n = mu_vol / exp(3.5 * ln(sigma_g)^2)
        median_n = mu_vol / math.exp(3.5 * ln_sig_g_sq)
        
        print(f"\n{' SAXS to Log-Normal Conversion (Exact) ':-^60}")
        print(f"Input (SAXS Vol): μ={mu_vol:.3f} nm, PD={pd_vol_pct:.1f}%")
        print(f"Result (Num)    : Median={median_n:.3f} nm, σg={sigma_g:.3f}")
        print("-" * 60)
        
        return cls.from_lognormal_params(median=median_n, sigma_g=sigma_g, amplitude=total_n)

    @staticmethod
    def _lognormal_model(x, amplitude, median, sigma_g):
        """Probability Density Function for Log-normal."""
        x = np.where(x <= 0, 1e-9, x)
        ln_sig_g = np.log(sigma_g)
        term1 = amplitude / (x * ln_sig_g * np.sqrt(2 * np.pi))
        term2 = np.exp(- (np.log(x / median))**2 / (2 * ln_sig_g**2))
        return term1 * term2
        
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
            dict: Dictionary containing amplitude, mean, sigma, FWHM, and CV (%).
            
        Raises:
            ValueError: If called before running the .fit() method.
        """
        if self.model_type == 'lognormal' and self._results_dict:
            return self._results_dict
        
        if self.params is None:
            raise ValueError("Fit has not been performed yet.")
        
        # Handle Gaussian logic (as per your original code)
        A, mu, sigma = self.params
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        cv = (sigma / mu) * 100
        return {
            "amplitude": A,
            "mean": mu,
            "sigma": sigma,
            "fwhm": fwhm,
            "cv_percentage": cv
        }

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
        from scipy.stats import norm, lognorm
        
        centerTitle("Summary of the distribution statistics")
        stats = self.results
        model = getattr(self, 'model_type', 'gaussian')

        # Basic Stats
        print(f"Model Type          : {model.upper()}")
        print(f"Amplitude           : {stats['amplitude']:.0f} particles")
        print(f"Average Size (Mean) : {stats['mean']:.3f} nm")
        print(f"Polydispersity (CV) : {stats['cv_percentage']:.2f}%")

        if model == 'lognormal':
            med, sg = stats['median'], stats['sigma_g']
            s_param = np.log(sg)
            
            # Dynamic calculation function for Lognormal
            def get_prob(low, high):
                return (lognorm.cdf(high, s=s_param, scale=med) - 
                        lognorm.cdf(low, s=s_param, scale=med)) * 100

            # Ranges
            r1 = (med/sg, med*sg)
            r2 = (med/(sg**2), med*(sg**2))
            r3 = (med/(sg**3), med*(sg**3))
            
            print(f"Geometric SD (σg)   : {sg:.3f}")
            print(f"Median Size         : {med:.3f} nm")
            print("-" * 40)
            print(f"Theoretical Population Coverage (Dynamic):")
            print(f"  Med */ 1σg ({r1[0]:>5.2f}-{r1[1]:<5.2f} nm) : {get_prob(*r1):.1f}%")
            print(f"  Med */ 2σg ({r2[0]:>5.2f}-{r2[1]:<5.2f} nm) : {get_prob(*r2):.1f}%")
            print(f"  Med */ 3σg ({r3[0]:>5.2f}-{r3[1]:<5.2f} nm) : {get_prob(*r3):.1f}%")
        
        else:
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
        """Calculates height relative to the peak (mu)."""
        mu = self.results['mean']
        sigma = self.results['sigma']
        z = (x_value - mu) / sigma
        return np.exp(-0.5 * z**2)
        
        
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
            - For Log-normal models, the lower limit is strictly clamped at 0.01 nm.
            - Synchronization: Updating the bins here will immediately change the 
              appearance of the histogram in the plot() method.
        """
        from scipy.stats import norm, lognorm

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
            # 2. Sinon, si c'est du Log-Normal, on prend l'amplitude stockée
            elif getattr(self, 'model_type', 'gaussian') == 'lognormal':
                total_n = self.results.get('amplitude', 1000)
            # 3. En dernier recours, on somme (cas des vraies données expérimentales)
            elif len(self.counts) > 0 and self.cov is not None and np.any(self.cov > 0):
                total_n = np.sum(self.counts)
            else:
                total_n = 1000
            
        # 1. Define coverage limits (±3.5 sigma)
        # For lognormal, we must ensure we don't go below or equal to zero
        limit_min = mu - 3.5 * sigma
        if getattr(self, 'model_type', 'gaussian') == 'lognormal':
            limit_min = max(0.01, limit_min)
        limit_max = mu + 3.5 * sigma
        
        # 2. Generate bin edges starting from mu to ensure symmetry (for Gaussian)
        # or just a consistent range for Log-normal
        right_edges = np.arange(mu + w/2, limit_max + w, w)
        left_edges = np.arange(mu - w/2, limit_min - w, -w)
        edges = np.sort(np.concatenate([left_edges, right_edges]))
        
        # --- 1. First pass: calculate data and sum of ratios for normalization ---
        bins_results = []
        total_ratio_sum = 0
        
        for i in range(len(edges) - 1):
            s1, s2 = edges[i], edges[i+1]
            bin_center = (s1 + s2) / 2
            
            if getattr(self, 'model_type', 'lognormal') == 'lognormal' and hasattr(self, '_lognormal_model'):
                # Lognormal probability and relative height
                s_param = np.log(stats['sigma_g'])
                prob = lognorm.cdf(s2, s=s_param, scale=stats['median']) - \
                       lognorm.cdf(s1, s=s_param, scale=stats['median'])
                peak_val = self._lognormal_model(stats['median'], 1, stats['median'], stats['sigma_g'])
                current_val = self._lognormal_model(bin_center, 1, stats['median'], stats['sigma_g'])
                ratio_to_peak = current_val / peak_val
            else:
                # Gaussian probability and relative height
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

    def get_proportions(self, target_sizes, bin_width_nm=None):
        """
        Calculate relative proportions and estimated counts for specific diameters.

        This method determines how specific particle sizes relate to the overall 
        distribution. It calculates the ratio of the probability density at the 
        target size relative to the peak (Ratio/Peak). 

        Critically, it also provides a 'Normalized' value (Norm. (1)). This is 
        calculated by summing the relative heights of all theoretical bins in 
        the distribution and dividing the target's ratio by this sum, allowing 
        for a discrete weight comparison.

        Args:
            target_sizes (float or array-like): The specific diameters (nm) to evaluate.
            bin_width_nm (float, optional): The reference bin width used to 
                calculate the total distribution sum for normalization. 
                Defaults to sigma.

        Returns:
            dict: A dictionary containing:
                - "sizes": The input target diameters.
                - "ratios": Probability at target relative to the peak (0 to 1).
                - "counts": Estimated number of particles at these specific sizes.
                - "norms": Normalized weight relative to the full binned distribution.
        """
        targets = np.atleast_1d(target_sizes)
        stats = self.results
        mu, sigma = stats['mean'], stats['sigma']
        
        # 1. Calculate the total_ratio_sum from the theoretical bins (for normalization)
        if bin_width_nm is None: bin_width_nm = sigma
        limit_min, limit_max = mu - 3.5 * sigma, mu + 3.5 * sigma
        if getattr(self, 'model_type', 'gaussian') == 'lognormal':
            limit_min = max(0.01, limit_min)
            
        r_edges = np.arange(mu + bin_width_nm/2, limit_max + bin_width_nm, bin_width_nm)
        l_edges = np.arange(mu - bin_width_nm/2, limit_min - bin_width_nm, -bin_width_nm)
        edges = np.sort(np.concatenate([l_edges, r_edges]))
        
        total_ratio_sum = 0
        for i in range(len(edges) - 1):
            bc = (edges[i] + edges[i+1]) / 2
            if getattr(self, 'model_type', 'gaussian') == 'lognormal':
                peak = self._lognormal_model(stats['median'], 1, stats['median'], stats['sigma_g'])
                total_ratio_sum += self._lognormal_model(bc, 1, stats['median'], stats['sigma_g']) / peak
            else:
                total_ratio_sum += np.exp(-0.5 * ((bc - mu) / sigma)**2)

        # 2. Calculate ratios for the specific targets
        if self.model_type == 'lognormal':
            peak_val = self._lognormal_model(stats['median'], 1, stats['median'], stats['sigma_g'])
            ratios = self._lognormal_model(targets, 1, stats['median'], stats['sigma_g']) / peak_val
        else:
            z = (targets - mu) / sigma
            ratios = np.exp(-0.5 * z**2)

        # 3. Normalize based on the distribution sum
        norm_values = ratios / total_ratio_sum if total_ratio_sum > 0 else 0
        norm_sum = ratios.sum()
        norm_relative = ratios / norm_sum if norm_sum > 0 else np.zeros_like(ratios)
        counts = ratios * (stats['amplitude'] if self.params is not None else 1000)
        
        return {
            "sizes": targets,
            "ratios": ratios,
            "counts": counts,
            "norms": norm_values,
            "norms_relative": norm_relative   #
        }
        
    def print_specific_proportions(self, target_sizes, labels=None):
        """
        Prints a formatted summary including the normalized distribution value.
        """
        data = self.get_proportions(target_sizes)
        
        centerTitle("Specific Diameter Proportions")
        # Added Norm. (1) column
        has_labels = labels is not None and len(labels) == len(data['sizes'])
    
        label_col = f"{'Label':<8} | " if has_labels else ""
        header = f"{label_col}{'Diameter (nm)':<15} | {'Ratio/Peak':<12} | {'Est. Count':<12} | {'Norm. (dist)':<14} | {'Norm. (1)':<12}"
        print(header)
        print("-" * len(header))
        
        for i in range(len(data['sizes'])):
            label_str = f"{labels[i]:<8} | " if has_labels else ""
            print(f"{label_str}"
                  f"{data['sizes'][i]:>12.2f} nm | "
                  f"{data['ratios'][i]:>10.3f}   | "
                  f"{data['counts'][i]:>10.0f}   | "
                  f"{data['norms'][i]:>12.4f}   | "
                  f"{data['norms_relative'][i]:>10.4f}")
        print("-" * len(header))
        
    def plot(self, title='Nanoparticle Size Distribution', color_histo="skyblue",
             color_gaussian="red", plot_histogram=True, highlight_sizes=None,
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
            color_gaussian (str): Color name or hex code for the Gaussian fit line.
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
        else:
            y_smooth = self._lognormal_model(x_smooth, *self.params) * scaling_w
            model_name = "Log-Normal"
        
        label_text = (f"{model_name} {'Fit' if scaling_w == 1.0 else 'Model'}:\n"
                      f"$\mu$ = {res['mean']:.2f} nm\n"
                      f"$\sigma$ = {res['sigma']:.2f} nm\n"
                      f"Polydispersity = {res['cv_percentage']:.1f}%")
        
        plt.plot(x_smooth, y_smooth, color=color_gaussian, lw=2, label=label_text)

        # Plot FWHM (Full Width at Half Maximum) indicator line
        plt.hlines(y=res['amplitude']/2 * scaling_w, 
                   xmin=res['mean'] - res['fwhm']/2, 
                   xmax=res['mean'] + res['fwhm']/2, 
                   colors='green', linestyles='--',
                   label=f"FWHM span ({res['fwhm']:.2f} nm)")

        # --- 5. Statistical Overlays ---
        # Vertical lines at +/- 1 and 2 sigma to visualize spread
        plt.axvline(x=mu - sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 1\sigma$ (Spread)")
        plt.axvline(x=mu + sigma, color='#3f8188', linestyle=':', lw=1.5)
        plt.axvline(x=mu - 2*sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 2\sigma$ (Spread)")
        plt.axvline(x=mu + 2*sigma, color='#3f8188', linestyle=':', lw=1.5)

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
                y_pos = (self._gaussian_model(size, *self.params) * scaling_w if self.model_type == 'gaussian'
                         else self._lognormal_model(size, *self.params) * scaling_w)
                
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