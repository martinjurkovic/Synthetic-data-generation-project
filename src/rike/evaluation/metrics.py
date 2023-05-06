import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn.metrics.pairwise import pairwise_distances
from sdmetrics.multi_table import CardinalityShapeSimilarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sdmetrics.utils import HyperTransformer
from sklearn.ensemble import RandomForestClassifier


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


def ks_test(original, synthetic, **kwargs):
    """
    Calculate the Kolmogorov-Smirnov test.
    """
    res = []
    for col in original.columns:
        statistic, _ = ks_2samp(original[col], synthetic[col])
        res.append(1 - statistic)

    return np.mean(res)


def chisquare_test(original, synthetic, nbins=10, **kwargs):
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


def js_divergence(original, synthetic, nbins=10, normalize=False, **kwargs):
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


def mean_max_discrepency(original, synthetic, kernel='linear', nbins=10, **kwargs):
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
            

def ml_efficiency(train_original, train_synthetic, test, validation=None, **kwargs):
    """
    Calculate the efficiency of the ML model.
    """
    pass


def discriminative_measure(original, synthetic, **kwargs):
    """
    Calculate the discriminative measure using a ML model.
    """
    pass


def distance_to_closest_record(original, synthetic, **kwargs):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan')
    return np.min(distances, axis=1)


def nearest_neighbour_distance_ratio(original, synthetic, **kwargs):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan', **kwargs)
    distances = np.sort(distances, axis=1)
    nearest = distances[:, 0]
    second_nearest = distances[:, 1]
    return nearest / second_nearest

# multi table metrics
def cardinality_similarity(tables_original, tables_synthetic, metadata, **kwargs):
    cardinality = CardinalityShapeSimilarity.compute_breakdown(tables_original, tables_synthetic, metadata)
    similarities = {}
    for table1, table2 in cardinality.keys():
        if table1 not in similarities:
            similarities[table1] = [cardinality[(table1, table2)]['score']]
        else:
            similarities[table1].append(cardinality[(table1, table2)]['score'])
        if table2 not in similarities:
            similarities[table2] = [cardinality[(table1, table2)]['score']]
        else:
            similarities[table2].append(cardinality[(table1, table2)]['score'])
    return similarities

# discriminative detection

def logistic_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=LogisticRegression(solver='lbfgs'), **kwargs)

def random_forest_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=RandomForestClassifier(), **kwargs)

def svm_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=SVC(gamma='auto'), **kwargs)

def discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=LogisticRegression(solver='lbfgs'), **kwargs):
    metadata = kwargs.get('metadata', None)

    if metadata is not None and 'primary_key' in metadata:
        transformed_original_train = original_train.drop(metadata['primary_key'], axis=1)
        transformed_synthetic_train = synthetic_train.drop(metadata['primary_key'], axis=1)
        transformed_original_test = original_test.drop(metadata['primary_key'], axis=1)
        transformed_synthetic_test = synthetic_test.drop(metadata['primary_key'], axis=1)
    else:
        transformed_original_train = original_train
        transformed_synthetic_train = synthetic_train
        transformed_original_test = original_test
        transformed_synthetic_test = synthetic_test

    ht = HyperTransformer()
    transformed_original_train = ht.fit_transform(transformed_original_train).to_numpy()
    transformed_original_test = ht.transform(transformed_original_test).to_numpy()
    transformed_synthetic_train = ht.transform(transformed_synthetic_train).to_numpy()
    transformed_synthetic_test = ht.transform(transformed_synthetic_test).to_numpy()
    
    X_train = np.concatenate([transformed_original_train, transformed_synthetic_train])
    X_test = np.concatenate([transformed_original_test, transformed_synthetic_test])
    y_train = np.hstack([
        np.ones(len(transformed_original_train)), np.zeros(len(transformed_synthetic_train))
    ])
    y_test = np.hstack([
        np.ones(len(transformed_original_test)), np.zeros(len(transformed_synthetic_test))
    ])

    clf = clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    return accuracy_score(y_test, y_pred)
