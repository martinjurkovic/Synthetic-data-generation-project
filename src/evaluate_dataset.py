import numpy as np
import pandas as pd
from rike.evaluation.metrics import ks_test, chisquare_test, mean_max_discrepency, js_divergence

import sdmetrics

df1 = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))

print(ks_test(df1, df2))
print(chisquare_test(df1, df2))
print(mean_max_discrepency(df1, df2))
print(js_divergence(df1, df2))

# tables_train, tables_test = utils.get_train_test_split(DATASET_NAME, k)
# metadata = generate_metadata.generate_metadata("biodegradability", tables_train)


# metrics = []
# scores = sdmetrics.compute_metrics(metrics, df1, df2, metadata=metadata)