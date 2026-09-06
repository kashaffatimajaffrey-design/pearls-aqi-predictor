# Experiments

Measurements taken before committing to a change, so the decision is evidence-led
rather than intuition-led.

## `exp_future_weather.py`

**Question.** How much would using *forecast* weather as a known-future covariate
improve the forecast, versus the current design which only sees weather as it is
now?

**Method.** We already hold a year of observed weather, so shifting it backwards
by 24/48/72 h hands the model a *perfect* weather forecast. That measures the
**upper bound** of the gain without a single new API call. A real Open-Meteo
forecast carries error, so the realised gain is some fraction of this.

**Result** (XGBoost, same chronological hold-out as the training pipeline):

| Horizon | RMSE now | RMSE ceiling | Gain | R² now | R² ceiling |
| --- | --- | --- | --- | --- | --- |
| +24 h | 5.34 | 5.01 | +6.2% | 0.695 | 0.732 |
| +48 h | 9.40 | 8.44 | **+10.2%** | 0.058 | 0.240 |
| +72 h | 10.32 | 9.44 | +8.5% | **−0.136** | **+0.050** |
| overall | 8.63 | 7.86 | +8.9% | 0.206 | 0.341 |

**Reading it.** Gains concentrate at the longer horizons, which is the expected
shape: at +24 h the lag features still carry most of the signal, whereas at
+48/72 h the current weather is stale and meteorology is what actually decides
the outcome.

The result that matters most is the +72 h R² crossing from **negative to
positive**. A negative R² means the model is doing worse than simply predicting
the mean — in variance terms the three-day forecast currently earns its keep only
through the AQI level, not through explaining variation. Known-future weather is
what changes that.

**Caveat.** Open-Meteo's archive weather is ERA5 reanalysis while its forecast
comes from a different NWP model, so a faithful implementation should train on
forecast-model history (`past_days` on the forecast endpoint) rather than mixing
the two, or the train/serve distributions will not match.

Run it with:

```bash
python -m experiments.exp_future_weather
```
