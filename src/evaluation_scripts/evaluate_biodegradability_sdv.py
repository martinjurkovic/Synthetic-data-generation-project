# %%
from rike.evaluation.metrics import ks_test, chisquare_test, mean_max_discrepency, js_divergence, logistic_detection
from rike.evaluation.report import generate_report
import json

DATASET_NAME = "biodegradability"
METHOD_NAME = "sdv"


# %%
single_table_metrics = [ks_test,
                        #logistic_detection
                        # chisquare_test,
                        # mean_max_discrepency,
                        # js_divergence,
                        ]

# %%
report = generate_report(DATASET_NAME, METHOD_NAME,
                         single_table_metrics=single_table_metrics, save_report=True)
# print formatted report dict
print(json.dumps(report, indent=4))

# %%
