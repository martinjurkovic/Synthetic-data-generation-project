import os

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
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sdmetrics.utils import HyperTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rike.utils import merge_children


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
        if "datetime" in str(original[col].dtype):
            original[col] = original[col].astype("int64")
            synthetic[col] = synthetic[col].astype("int64") 

        if len(original[col].unique()) < 5 or original[col].dtype == 'object':  # categorical
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
    metadata = kwargs.get("metadata", None)
    orig = original.copy().dropna()
    synth = synthetic.copy().dropna()
    orig.drop(metadata['primary_key'], axis=1, inplace=True)
    synth.drop(metadata['primary_key'], axis=1, inplace=True)

    res = []
    for col in orig.columns:
        if orig[col].dtype == "object":
            continue
        orig_col = orig[col]
        synth_col = synth[col]
        orig_col = orig_col[~orig_col.isna()].values
        synth_col = synth_col[~synth_col.isna()].values
        statistic, _ = ks_2samp(orig_col, synth_col)
        res.append(1 - statistic)
    if len(res):
        return np.mean(res)
    else:
        return -1


def chisquare_test(original, synthetic, nbins=10, **kwargs):
    """
    Calculate the Chi-Square test.
    """
    metadata = kwargs.get("metadata", None)

    res = []
    orig = original.copy().dropna()
    synth = synthetic.copy().dropna()

    orig.drop(metadata['primary_key'], axis=1, inplace=True)
    synth.drop(metadata['primary_key'], axis=1, inplace=True)

    freqs = get_frequency(
        orig, synth, nbins=nbins
    )

    for col in orig.columns:
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
    metadata = kwargs.get("metadata", None)

    orig = original.copy().dropna()
    synth = synthetic.copy().dropna()

    orig.drop(metadata['primary_key'], axis=1, inplace=True)
    synth.drop(metadata['primary_key'], axis=1, inplace=True)


    for col in orig.columns:
        if "datetime" in str(orig[col].dtype):
            orig[col] = orig[col].astype("int64")
            synth[col] = synth[col].astype("int64")
            maximum = orig[col].max()
            orig[col] = orig[col] / maximum
            synth[col] = synth[col] / maximum
        local_bins = min(nbins, len(orig[col].unique()))
        if orig[col].dtype == "object":
            # compute histogram bins for categorical features
            # without pd.cut as it does not support categorical data
            original_bin = orig[col].value_counts(dropna=False, normalize=normalize)
            synthetic_bin = synth[col].value_counts(dropna=False, normalize=normalize)
        else:
            original_bin, gt_bins = pd.cut(orig[col], bins=local_bins, retbins=True)
            synthetic_bin = pd.cut(synth[col], bins=gt_bins)
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
    # preprocess the values
    metadata = kwargs.get("metadata", None)
    orig = original.copy()
    synth = synthetic.copy()

    orig.drop(metadata['primary_key'], axis=1, inplace=True)
    synth.drop(metadata['primary_key'], axis=1, inplace=True)
    for col in orig.columns:
        if orig[col].dtype == "object":
            orig.drop(col, axis=1, inplace=True)
            synth.drop(col, axis=1, inplace=True)
        elif "datetime" in str(orig[col].dtype):
            orig[col] = orig[col].astype("int64")
            synth[col] = synth[col].astype("int64")
            maximum = orig[col].max()
            orig[col] = orig[col] / maximum
            synth[col] = synth[col] / maximum

    
    if kernel == "linear":
        """
        MMD using linear kernel (i.e., k(x,y) = <x,y>)
        """
        delta = orig.mean(axis=0) - synth.mean(axis=0)
        #delta = delta_df.values

        score = delta.dot(delta.T)
    elif kernel == "rbf":
        """
        MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        """
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(
            orig.reshape(len(original), -1),
            synth.reshape(len(original), -1),
            gamma,
        )
        YY = metrics.pairwise.rbf_kernel(
            synth.reshape(len(synthetic), -1),
            synth.reshape(len(synthetic), -1),
            gamma,
        )
        XY = metrics.pairwise.rbf_kernel(
            orig.reshape(len(original), -1),
            synth.reshape(len(synthetic), -1),
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
            orig.reshape(len(original), -1),
            orig.reshape(len(original), -1),
            degree,
            gamma,
            coef0,
        )
        YY = metrics.pairwise.polynomial_kernel(
            synth.numpy().reshape(len(synthetic), -1),
            synth.numpy().reshape(len(synthetic), -1),
            degree,
            gamma,
            coef0,
        )
        XY = metrics.pairwise.polynomial_kernel(
            orig.reshape(len(original), -1),
            synth.numpy().reshape(len(synthetic), -1),
            degree,
            gamma,
            coef0,
        )
        score = XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        raise ValueError(f"Unsupported kernel {kernel}")
    
    return score


def calculate_statistical_metrics(original, synthetic, **kwargs):
    ks = ks_test(original, synthetic, **kwargs)
    js = js_divergence(original, synthetic, **kwargs)
    mm = mean_max_discrepency(original, synthetic, **kwargs)
    ch = chisquare_test(original, synthetic, **kwargs)
    return [ks, js, mm, ch]


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
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=SVC(gamma='auto', probability=True), **kwargs)


def knn_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=KNeighborsClassifier(), **kwargs)


def mlp_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100, ))
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes), **kwargs)


def xgboost_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf = xgb.XGBClassifier(), **kwargs)


def discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=LogisticRegression(solver='lbfgs'), **kwargs):
    save_path = kwargs.get('save_path', None)
    metadata = kwargs.get('metadata', None)
    synthetic_ids = synthetic_test.get(metadata['primary_key'])
    original_ids =  original_test.get(metadata['primary_key'])
    transformed_original_train = original_train.copy()
    transformed_synthetic_train = synthetic_train.copy()
    transformed_original_test = original_test.copy()
    transformed_synthetic_test = synthetic_test.copy()
    if metadata is not None and 'primary_key' in metadata:
        transformed_original_train = transformed_original_train.drop(metadata['primary_key'], axis=1)
        transformed_synthetic_train = transformed_synthetic_train.drop(metadata['primary_key'], axis=1)
        transformed_synthetic_test = synthetic_test.drop(metadata['primary_key'], axis=1)
        transformed_original_test = transformed_original_test.drop(metadata['primary_key'], axis=1)
    

    ht = HyperTransformer()
    transformed_original_train = ht.fit_transform(transformed_original_train)
    columns = transformed_original_train.columns
    transformed_original_train = transformed_original_train.to_numpy()
    transformed_original_test = ht.transform(transformed_original_test).to_numpy()
    transformed_synthetic_train = ht.transform(transformed_synthetic_train).to_numpy()
    transformed_synthetic_test = ht.transform(transformed_synthetic_test).to_numpy()
    
    X_train = np.concatenate([transformed_original_train, transformed_synthetic_train])
    # X_test = transformed_synthetic_test 
    X_test = np.concatenate([transformed_original_test, transformed_synthetic_test])
    y_train = np.hstack([
        np.ones(len(transformed_original_train)), np.zeros(len(transformed_synthetic_train))
    ])
    # y_test = np.zeros(len(transformed_synthetic_test))
    y_test = np.hstack([
        np.ones(len(transformed_original_test)), np.zeros(len(transformed_synthetic_test))
    ])

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    if save_path is not None:
        # save probabilities
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({
            'id': np.hstack([original_ids, synthetic_ids]),
            'prob_is_fake': probs[:, 1],
            'y_pred': y_pred,
            'y_true': y_test
        })
        df.to_csv(save_path, index=False)

        # save feature importance
        # TODO: convert to wrapper
        if hasattr(clf, 'feature_importances_'):
            feature_importance_path = save_path.replace('probabilities', 'feature_importance')
            os.makedirs(os.path.dirname(feature_importance_path), exist_ok=True)
            df = pd.DataFrame({
                'feature': columns,
                'importance': clf.feature_importances_
            })
            df.sort_values(by='importance', ascending=False, inplace=True)
            df.to_csv(feature_importance_path, index=False)
    #return log_loss(y_test, probs)
    return [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), log_loss(y_test, probs)]


def parent_child_logistic_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=LogisticRegression(solver='lbfgs'), **kwargs)


def parent_child_random_forest_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=RandomForestClassifier(), **kwargs)


def parent_child_knn_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=KNeighborsClassifier(), **kwargs)


def parent_child_mlp_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100, ))
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes), **kwargs)


def parent_child_svm_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=SVC(probability=True), **kwargs)


def parent_child_xgb_detection(original_test, synthetic_test, original_train, synthetic_train, **kwargs):
    return parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=xgb.XGBClassifier(), **kwargs)

   
def parent_child_discriminative_detection(original_test, synthetic_test, original_train, synthetic_train, clf=LogisticRegression(solver='lbfgs'), **kwargs):
    metadata = kwargs.get('metadata', None)
    root_table = kwargs.get('root_table', None)
    save_path = kwargs.get('save_path', None)
    # join parent and child tables based on the metadata
    original_train = merge_children(original_train, metadata, root_table)
    synthetic_train = merge_children(synthetic_train, metadata, root_table)
    original_test = merge_children(original_test, metadata, root_table)
    synthetic_test = merge_children(synthetic_test, metadata, root_table)

    # drop all foreign and primary keys
    primary_key = metadata.to_dict()['tables'][root_table]['primary_key']
    synthetic_ids = synthetic_test.get(primary_key)
    original_ids =  original_test.get(primary_key)
    for table in metadata.to_dict()['tables'].keys():
        for field, values in metadata.to_dict()['tables'][table].items():
            if 'ref' in values:
                foreign_key = field
                for column in original_train.columns():
                    if foreign_key in column:
                        original_train.drop(column, axis=1, inplace=True)
                        synthetic_train.drop(column, axis=1, inplace=True)
                        original_test.drop(column, axis=1, inplace=True)
                        synthetic_test.drop(column, axis=1, inplace=True)
        pk = metadata.get_primary_key(table)
        for column in original_train.columns:
            if pk in column:
                original_train.drop(column, axis=1, inplace=True)
                synthetic_train.drop(column, axis=1, inplace=True)
                original_test.drop(column, axis=1, inplace=True)
                synthetic_test.drop(column, axis=1, inplace=True)

    ht = HyperTransformer()
    transformed_original_train = ht.fit_transform(original_train)
    columns = transformed_original_train.columns
    transformed_original_train = transformed_original_train.to_numpy()
    transformed_original_test = ht.transform(original_test).to_numpy()
    transformed_synthetic_train = ht.transform(synthetic_train).to_numpy()
    transformed_synthetic_test = ht.transform(synthetic_test).to_numpy()

    X_train = np.concatenate([transformed_original_train, transformed_synthetic_train])
    # X_test = transformed_synthetic_test 
    X_test = np.concatenate([transformed_original_test, transformed_synthetic_test])
    y_train = np.hstack([
        np.ones(len(transformed_original_train)), np.zeros(len(transformed_synthetic_train))
    ])
    # y_test = np.zeros(len(transformed_synthetic_test))
    y_test = np.hstack([
        np.ones(len(transformed_original_test)), np.zeros(len(transformed_synthetic_test))
    ])
    # construct the model pipeline
    # with standard scaler
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    if save_path is not None:
        # save probabilities
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({
            'id': np.hstack([original_ids, synthetic_ids]),
            'prob_is_fake': probs[:, 1],
            'y_pred': y_pred,
            'y_true': y_test
        })
        df.to_csv(save_path, index=False)

        # save feature importance
        # TODO: convert to wrapper
        if hasattr(clf, 'feature_importances_'):
            feature_importance_path = save_path.replace('probabilities', 'feature_importance')
            os.makedirs(os.path.dirname(feature_importance_path), exist_ok=True)
            df = pd.DataFrame({
                'feature': columns,
                'importance': clf.feature_importances_
            })
            df.sort_values(by='importance', ascending=False, inplace=True)
            df.to_csv(feature_importance_path, index=False)
    return [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), log_loss(y_test, probs)]