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

---

## `exp_sequence_variance.py`

**Question.** Is the encoder-decoder LSTM actually better than XGBoost, or was
the first favourable run luck?

**Why it was needed.** A single run gave the future-weather variant RMSE 8.50
against 9.80 without it — an apparently decisive win. Rerunning identical code
gave 9.72 and 9.62, **reversing the ordering**. Neural training here is
stochastic (weight init, dropout, CPU reduction order), so one run is not
evidence. `tf.random.set_seed` does not fully determinise it.

**Method.** Both variants trained across 5 seeds, scored on the same hold-out.

**Result.**

| Variant | mean RMSE | std | min | max |
| --- | --- | --- | --- | --- |
| Sequence LSTM + future weather | 8.885 | 0.890 | 8.116 | 10.076 |
| Sequence LSTM, no future weather | 10.772 | 1.685 | 9.307 | 12.869 |
| **XGBoost** (deterministic) | **8.628** | 0.000 | — | — |

### Two conclusions, pulling in opposite directions

**1. XGBoost still wins — the LSTM does not ship.** Its mean (8.885) is worse
than XGBoost's 8.628. It beat XGBoost on 3 of 5 seeds, which is exactly why a
single run is misleading: quoting the 8.116 seed would be seed-shopping. The
run-to-run spread (std 0.89, and 1.69 without future weather) is larger than the
gap being argued about. On 8k rows these networks are simply unstable.

**2. But the future-weather signal is real and large.** +1.887 RMSE mean
improvement, Cohen's d = 1.40, consistent in direction across every seed pair.
With n=5 that is indicative rather than conclusive, but it corroborates the
independent perfect-prog measurement in `exp_future_weather.py` (+8.9% for
XGBoost).

### The actual takeaway

**The information is valuable; the architecture is not.** Two independent
experiments agree that knowing future weather matters, and both say the LSTM is
not the way to consume it. The best number anywhere in this project is
**XGBoost + future weather at RMSE 7.862** — better than plain XGBoost (8.628)
and better than the sequence LSTM's mean (8.885).

So the recommendation is to add forecast-weather columns to the existing tabular
feature set, not to ship a sequence model.

### Caveat on all future-weather numbers

These use *observed* weather shifted backwards — perfect foresight ("perfect
prog"). It is an upper bound, not a deployable figure. A real Open-Meteo
forecast carries error, so expect a fraction of the gain.

Run it with:

```bash
python -m experiments.exp_sequence_variance
```
