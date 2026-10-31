# New grouping and repurchase model: unified historical playback

The new K-means 4 group and the new phase five selected 14 features XGBoost have been loaded into the same model package. The models are only fitted using the original training period; the scoring and grouping results of the validation period are reproduced one by one with the upstream products.
2011, September-November have been viewed in earlier projects, the following examination is a retrospective diagnosis and cannot be called a new independent test. Repurchase scores are not calibrated for probability and are only used for relative ranking.

| method | AP | Top 20% accuracy | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: |
| XGBoost (base_8_plus_trend_and_rhythm) | 0.6201 | 66.22% | 0.7204 | 0.1928 |
| LR (base_8) | 0.6064 | 63.96% | 0.7100 | 0.1969 |

Check monthly:
| month | method | AP | Top 20% accuracy |
| --- | --- | ---: | ---: |
| 2011-09-01 | XGBoost | 0.6205 | 64.98% |
| 2011-10-01 | XGBoost | 0.5740 | 60.62% |
| 2011-11-01 | XGBoost | 0.6636 | 72.41% |
| 2011-09-01 | LR | 0.6072 | 64.08% |
| 2011-10-01 | LR | 0.5562 | 56.85% |
| 2011-11-01 | LR | 0.6529 | 70.38% |

XGBoost relative to basic logistic regression: AP difference + 0.0137, Top 20% accuracy difference + 2.25 percentage points.
This illustrates the ranking differences on the same batch of historical records and does not infer that reaching customers increases revenue.

Run: `python -B -m stage6.integrate_reanalysis`; Page: `python -m streamlit run app/dashboard.py`. Item-by-item scoring In `test_predictions.csv`, dashboard export does not include post-event tags.
