# Directional optimization of repurchase model: forward calibration experiment

Fixed the 8 features of group A and the LightGBM parameters selected in stage five; before each month's prediction, first set aside the last three fully annotated monthly snapshots for calibration, and then use earlier months to fit the classifier. Compare raw score, recent 1 / 2 / 3 months sigmoid and recent 3 months isotonic.

2011, 3 – 5 month selection calibration method, June-August confirmation; September-November only for post-diagnosis. Existing Phase Six models viewed September-Novemberly results, so these months are no longer unviewed tests for the entire project.

Selection method: **sigmoid_ 2 m**; Confirmation threshold: **Passed**; Confirmation period Brier wins 2 / 3 months. Sorting threshold with existing models: **Failed**.

| stage | method | AP (monthly average) | Top 20% accuracy (monthly average) | Brier (monthly average) | Absolute mean calibration difference (monthly average) | ECE- 10 (monthly average) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tuning | raw | 0.5417 | 51.94% | 0.2151 | 25.09% | 25.09% |
| tuning | sigmoid_3m | 0.5417 | 51.94% | 0.1456 | 4.88% | 5.68% |
| tuning | sigmoid_2m | 0.5417 | 51.94% | 0.1440 | 3.07% | 3.99% |
| confirmation | raw | 0.5812 | 58.57% | 0.1863 | 15.98% | 15.98% |
| confirmation | sigmoid_3m | 0.5812 | 58.57% | 0.1612 | 3.21% | 3.73% |
| confirmation | sigmoid_2m | 0.5812 | 58.57% | 0.1607 | 2.65% | 3.30% |
| retrospective_test | raw | 0.6177 | 66.05% | 0.2062 | 13.20% | 13.20% |
| retrospective_test | sigmoid_3m | 0.6177 | 66.05% | 0.1900 | 4.93% | 5.45% |
| retrospective_test | sigmoid_2m | 0.6177 | 66.05% | 0.1895 | 4.35% | 5.04% |
| incumbent_validation | stage6_incumbent | 0.5955 | 60.30% | 0.1566 | 1.03% | 2.45% |
| incumbent_retrospective_test | stage6_incumbent | 0.6171 | 65.99% | 0.1920 | 6.15% | 6.57% |

AP and Top 20% accuracy are used for guard ranking; Brier, mean calibration difference and ten-bin ECE are used for checking probabilities. All calibrations use only 30 day labels that have ended before the target month.
The calibrator for the existing stage six validation score was fitted on the entire June-Augustly validation set; its validation period Brier therefore does not serve as a strict forward calibration control. Here only its AP and Top 20% precision rates are used as ranking gatekeeping indicators.
Compared with the existing model, the monthly average AP of the new scheme during the confirmation period is lower than 0.0143, and the monthly average accuracy of Top 20% is 1.73% lower, exceeding the 0.005 tolerance used in this experiment, so the existing model will not be replaced. 9 – 11 Monthly post-mortem diagnosis does not change this decision.
Existing stage 6 /outputs/final models and reports remain intact; this experiment does not automatically replace online or dashboard scoring.
Complete monthly results, item-by-item ratings and fixed rules can be found in the same directories CSV and decision.json.
