import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn.metrics.pairwise import pairwise_distances

def get_frequency(
    original: pd.DataFrame, synthetic: pd.DataFrame, nbins: int = 10
) -> dict:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in original.columns:
        local_bins = min(nbins, len(original[col].unique()))

        if len(original[col].unique()) < 5:  # categorical
            gt = (original[col].value_counts() / len(original)).to_dict()
            synth = (synthetic[col].value_counts() / len(synthetic)).to_dict()
        else:
            gt_vals, bins = np.histogram(original[col], bins=local_bins)
            synth_vals, _ = np.histogram(synthetic[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res


def ks_test(original, synthetic):
    """
    Calculate the Kolmogorov-Smirnov test.
    """
    res = []
    for col in original.columns:
        statistic, _ = ks_2samp(original[col], synthetic[col])
        res.append(1 - statistic)

    return np.mean(res)


def chisquare_test(original, synthetic, nbins=10):
    """
    Calculate the Chi-Square test.
    """
    res = []
    freqs = get_frequency(
        original, synthetic, nbins=nbins
    )

    for col in original.columns:
        gt_freq, synth_freq = freqs[col]
        try:
            _, pvalue = chisquare(gt_freq, synth_freq)
            if np.isnan(pvalue):
                pvalue = 0
        except BaseException:
            pvalue = 0
        res.append(pvalue)
    
    return np.mean(res)


def js_divergence(original, synthetic, nbins=10, normalize=False):
    stats_gt = {}
    stats_syn = {}
    stats = {}

    for col in original.columns:
        local_bins = min(nbins, len(original[col].unique()))
        original_bin, gt_bins = pd.cut(original[col], bins=local_bins, retbins=True)
        synthetic_bin = pd.cut(synthetic[col], bins=gt_bins)
        stats_gt[col], stats_syn[col] = original_bin.value_counts(
            dropna=False, normalize=normalize
        ).align(
            synthetic_bin.value_counts(dropna=False, normalize=normalize),
            join="outer",
            axis=0,
            fill_value=0,
        )
        stats_gt[col] += 1
        stats_syn[col] += 1

        stats[col] = jensenshannon(stats_gt[col], stats_syn[col])
        if np.isnan(stats[col]):
            raise RuntimeError("NaNs in prediction")
        
    return sum(stats.values()) / len(stats.keys())


def mean_max_discrepency(original, synthetic, kernel='linear', nbins=10):
    if kernel == "linear":
        """
        MMD using linear kernel (i.e., k(x,y) = <x,y>)
        """
        delta_df = original.mean(axis=0) - synthetic.mean(axis=0)
        delta = delta_df.values

        score = delta.dot(delta.T)
    elif kernel == "rbf":
        """
        MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        """
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(
            original.numpy().reshape(len(original), -1),
            original.numpy().reshape(len(original), -1),
            gamma,
        )
        YY = metrics.pairwise.rbf_kernel(
            synthetic.numpy().reshape(len(synthetic), -1),
            synthetic.numpy().reshape(len(synthetic), -1),
            gamma,
        )
        XY = metrics.pairwise.rbf_kernel(
            original.numpy().reshape(len(original), -1),
            synthetic.numpy().reshape(len(synthetic), -1),
            gamma,
        )
        score = XX.mean() + YY.mean() - 2 * XY.mean()
    elif kernel == "polynomial":
        """
        MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        """
        degree = 2
        gamma = 1
        coef0 = 0
        XX = metrics.pairwise.polynomial_kernel(
            original.numpy().reshape(len(original), -1),
            original.numpy().reshape(len(original), -1),
            degree,
            gamma,
            coef0,
        )
        YY = metrics.pairwise.polynomial_kernel(
            synthetic.numpy().reshape(len(synthetic), -1),
            synthetic.numpy().reshape(len(synthetic), -1),
            degree,
            gamma,
            coef0,
        )
        XY = metrics.pairwise.polynomial_kernel(
            original.numpy().reshape(len(original), -1),
            synthetic.numpy().reshape(len(synthetic), -1),
            degree,
            gamma,
            coef0,
        )
        score = XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        raise ValueError(f"Unsupported kernel {kernel}")
    
    return score
            

def ml_efficiency(train_original, train_synthetic, test, validation=None):
    """
    Calculate the efficiency of the ML model.
    """
    pass


def discriminative_measure(original, synthetic):
    """
    Calculate the discriminative measure using a ML model.
    """
    pass

def distance_to_closest_record(original, synthetic):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan')
    return np.min(distances, axis=1)

def nearest_neighbour_distance_ratio(original, synthetic):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan')
    distances = np.sort(distances, axis=1)
    nearest = distances[:, 0]
    second_nearest = distances[:, 1]
    return nearest / second_nearest