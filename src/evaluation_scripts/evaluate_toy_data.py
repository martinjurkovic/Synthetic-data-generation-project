import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rike.evaluation.metrics import ks_test, chisquare_test, mean_max_discrepency, js_divergence, distance_to_closest_record, nearest_neighbour_distance_ratio

import sdmetrics

df1 = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
df3 = pd.DataFrame(np.random.randn(1000, 4) * 2, columns=list('ABCD'))
df4 = df2.copy()
df4[:500] = df1[:500]


dcr = distance_to_closest_record(df1, df2)
dcr_same = distance_to_closest_record(df1, df1)
dcr_bad = distance_to_closest_record(df1, df3)
dcr_copied = distance_to_closest_record(df1, df4)

nndr = nearest_neighbour_distance_ratio(df1, df2)
nndr_same = nearest_neighbour_distance_ratio(df1, df1)
nndr_bad = nearest_neighbour_distance_ratio(df1, df3)
nndr_copied = nearest_neighbour_distance_ratio(df1, df4)

plt.hist(dcr, label='good')
plt.hist(dcr_bad, label='bad', alpha=0.5)
plt.hist(dcr_same, label='same', alpha=0.5)
plt.hist(dcr_copied, label='copied', alpha=0.5)
plt.legend()
plt.show()

plt.hist(nndr, label='good')
plt.hist(nndr_bad, label='bad', alpha=0.5)
plt.hist(nndr_same, label='same', alpha=0.5)
plt.hist(nndr_copied, label='copied', alpha=0.5)
plt.legend()
plt.show()

# tables_train, tables_test = utils.get_train_test_split(DATASET_NAME, k)
# metadata = generate_metadata.generate_metadata("biodegradability", tables_train)


# metrics = []
# scores = sdmetrics.compute_metrics(metrics, df1, df2, metadata=metadata)