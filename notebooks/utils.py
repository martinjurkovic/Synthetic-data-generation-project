import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sdmetrics.reports import utils

def read_tables(path, split_by="-", name_index=-1, **kwargs):
    tables = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            table_name = file[:-4].split(split_by)[name_index]
            table = pd.read_csv(
                path + f'{"/" if path[-1] != "/" else ""}' + file, **kwargs)
            tables[table_name] = table
    return tables


def plot_all_columns(real_data, synthetic_data, metadata):
    for table_name, table in real_data.items():
        for column_name in table.columns:
            try:
                fig = utils.get_column_plot(
                    real_data=real_data[table_name],
                    synthetic_data=synthetic_data[table_name],
                    column_name=column_name,
                    metadata=metadata['tables'][table_name]
                )
                fig.show()
            except Exception as e:
                print(f"Could not plot {table_name}.{column_name}")
                print(e)


def plot_parent_child_joint_distribution(df_parent, df_child, parent_column, child_column, metadata, ax=None, kind='hist', samp_size=0.5, binrange=None, nbins=20, **kwargs):
    sns.set_theme()
    # TODO key is hardcoded as metadata?
    key = metadata
    df1 = df_parent[[parent_column, key]]
    df2 = df_child[[child_column, key]]
    df_joint = df1.merge(df2, right_on=key, how='outer', left_on=key).dropna()
    df = df_joint.sample(frac=samp_size)
    child = df[child_column].values
    parent = df[parent_column].values
    binsparent = min(nbins, np.unique(parent).shape[0])
    binschild = min(nbins, np.unique(child).shape[0])
    if kind == 'hist':
        # sort by both parent and child
        df = df.sort_values(by=[parent_column, child_column])
        sns.histplot(df, x=parent_column, y=child_column, ax=ax, bins = (binsparent, binschild), binrange=binrange)
    elif kind == 'kde':
        sns.kdeplot(df, x=parent_column, y=child_column, ax=ax) # shade=True, shade_lowest=False, 
    else:
        if 'xedges' in kwargs and 'yedges' in kwargs:
            xedges = kwargs['xedges']
            yedges = kwargs['yedges']
        else:
            _, xedges = np.histogram(parent, bins=binsparent)
            _, yedges = np.histogram(child, bins=binschild)
        H, xedges, yedges = np.histogram2d(parent, child, bins=(xedges, yedges))
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H.T, cmap='Blues')
        return xedges, yedges


def plot_real_vs_synthetic_parent_child(real, synthetic, parent, child, col_parent, col_child, kind = 'hist', samp_size=0.5):
    sns.set_theme()
    todo = 'store'
    fig, axes = plt.subplots(1, 2)
    if kind == 'hist':
        if real[parent][col_parent].dtype == 'object' or synthetic[parent][col_parent].dtype == 'object':
            binrange = None
        else:
            rp_max = real[parent][col_parent].max()
            sp_max = synthetic[parent][col_parent].max()
            rc_max = real[child][col_child].max()
            rs_max = synthetic[child][col_child].max()
            rp_min = real[parent][col_parent].min()
            sp_min = synthetic[parent][col_parent].min()
            rc_min = real[child][col_child].min()
            rs_min = synthetic[child][col_child].min()
            binrange = (min(rp_min, sp_min), max(rp_max, sp_max)), (min(rc_min, rs_min), max(rc_max, rs_max))
        plot_parent_child_joint_distribution(real[parent], real[child], col_parent, col_child, todo, ax = axes[0], kind=kind, binrange=binrange, samp_size=samp_size)
        plot_parent_child_joint_distribution(synthetic[parent], synthetic[child], col_parent, col_child, todo, ax = axes[1], kind=kind, binrange=binrange, samp_size=samp_size)
        # set axes to be the same
        x1, x2, y1, y2 = axes[0].axis()
        x3, x4, y3, y4 = axes[1].axis()
        xmin = min(x1, x3)
        xmax = max(x2, x4)
        ymin = min(y1, y3)
        ymax = max(y2, y4)
        axes[0].axis((xmin, xmax, ymin, ymax))
        axes[1].axis((xmin, xmax, ymin, ymax))
    elif  kind == 'kde':
        plot_parent_child_joint_distribution(real[parent], real[child], col_parent, col_child, todo, ax = axes[0], kind=kind, samp_size=samp_size)
        plot_parent_child_joint_distribution(synthetic[parent], synthetic[child], col_parent, col_child, todo, ax = axes[1], kind=kind, samp_size=samp_size)
    elif kind == 'hm':
        xedges, yedges = plot_parent_child_joint_distribution(real[parent], real[child], col_parent, col_child, todo, ax = axes[0], kind=kind, samp_size=samp_size)
        plot_parent_child_joint_distribution(synthetic[parent], synthetic[child], col_parent, col_child, todo, ax = axes[1], kind=kind, xedges=xedges, yedges=yedges, samp_size=samp_size)
    else:
        raise ValueError(f'Invalid kind: {kind}')
    axes[0].set_title('Real Data')
    axes[1].set_title('Synthetic Data')
    fig.suptitle(f'Parent-Child Joint Distribution: {col_parent} vs {col_child}')
    fig.tight_layout()