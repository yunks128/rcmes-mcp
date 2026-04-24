# Example Prompts — Advanced Spatiotemporal Analysis

These prompts demo the Pillar 1 (anomaly detection) and Pillar 2 (ensembles & scenarios) tools added to the RCMES chat. Paste any of them into the web UI at `http://localhost:8502` or the `rcmes-chat` CLI.

---

## 1. Find extreme heat events in California

> Find extreme heat events in California from 2050–2055 under SSP5-8.5. Use ACCESS-CM2 and a baseline of 1980–2010. Run anomaly detection at 2σ for at least 3 days, then show a Hovmöller diagram.

**Tool chain**

```
load_climate_data(tasmax, ACCESS-CM2, ssp585, 2050–2055, California bbox)
  → calculate_standardized_anomaly(baseline 1980-01-01 → 2010-12-31)
  → detect_extreme_events(sigma_threshold=2.0, min_duration_days=3, direction="positive")
  → generate_hovmoller(average_over="lon")
```

---

## 2. Dominant variability pattern over the tropical Pacific (EOF)

> Load tasmax for the tropical Pacific (lat -20 to 20, lon 150 to 280) under historical 1980–2014 with CESM2. Compute anomalies and run an EOF with 3 modes — then map the leading spatial pattern and plot the PC1 time series.

**Tool chain**

```
load_climate_data(tasmax, CESM2, historical, 1980–2014, lat -20:20, lon 150:280)
  → calculate_anomaly(baseline 1980-01-01 → 2014-12-31)
  → calculate_eof(n_modes=3, detrend=True)
  → generate_map(spatial_dataset_id, colormap="RdBu_r")
  → generate_timeseries_plot([pc_dataset_id])
```

---

## 3. Multi-model ensemble with model agreement (IPCC-style)

> Build an ensemble of 5 models (ACCESS-CM2, GFDL-ESM4, MRI-ESM2-0, CESM2, MPI-ESM1-2-HR) for tasmax over India 2070–2099 under SSP5-8.5. Calculate ensemble statistics with a 1980–2010 baseline. Map the ensemble-mean change AND the agreement map side-by-side.

**Tool chain**

```
load_multi_model_ensemble(tasmax, [5 models], ssp585, 2070–2099, India bbox)
  → calculate_ensemble_statistics(baseline 1980-01-01 → 2010-12-31)
  → generate_map(ensemble_mean_id)
  → generate_map(agreement_id, colormap="RdBu_r", vmin=0, vmax=1)
```

---

## 4. Scenario comparison fan chart

> Compare scenarios SSP1-2.6, SSP2-4.5, and SSP5-8.5 for tas over Texas from 2015 to 2099 with MRI-ESM2-0. Show me a fan chart and map the SSP5-8.5 minus SSP1-2.6 difference.

**Tool chain**

```
compare_scenarios(tas, MRI-ESM2-0, [ssp126, ssp245, ssp585], 2015–2099, Texas bbox)
  → generate_scenario_fan_chart(per_scenario mapping, smooth_window=12)
  → generate_map(differences[ssp585 vs ssp126].dataset_id, colormap="RdBu_r")
```

---

## 5. Time of emergence over Europe

> Load tasmax for Europe (lat 35–70, lon -10 to 40) from 1950 to 2099 under SSP5-8.5 with GFDL-ESM4. Compute time-of-emergence using 1950–2000 as baseline and a 2σ threshold. Map when each cell experiences emergence.

**Tool chain**

```
load_climate_data(tasmax, GFDL-ESM4, ssp585, 1950–2099, Europe bbox)
  → calculate_time_of_emergence(baseline 1950-01-01 → 2000-12-31, sigma_threshold=2.0, rolling_years=20)
  → generate_map(emergence_year_id, colormap="viridis")
```

NaN cells = signal never emerges within the dataset window.

---

## 6. Detect extreme drought events

> For Brazil 2030–2060 under SSP3-7.0, load precipitation with ACCESS-CM2. Convert to mm/day. Standardize against the 1980–2010 baseline and detect negative extreme events at -1.5σ lasting 14+ days.

**Tool chain**

```
load_climate_data(pr, ACCESS-CM2, ssp370, 2030–2060, Brazil bbox)
  → convert_units(target_unit="mm/day")
  → calculate_standardized_anomaly(baseline 1980-01-01 → 2010-12-31)
  → detect_extreme_events(sigma_threshold=1.5, min_duration_days=14, direction="negative")
```

---

## Notes on usage

- **Standardize before detecting events.** `detect_extreme_events` expects a z-score field. Use `calculate_standardized_anomaly` (not the plain `calculate_anomaly`) before it.
- **Coarsen before EOF.** Full-globe daily fields are RAM-heavy; subset spatially and/or `temporal_resample` to monthly first.
- **Ensemble cap is 10 models** — pre-pick the most representative ones. Loading and stacking happens sequentially, so 10 models ≈ 10× a single load.
- **Agreement map only appears** when `baseline_start`/`baseline_end` are passed to `calculate_ensemble_statistics`.
- **Time-of-emergence requires a long input window** — the algorithm needs both a stable baseline and a future signal, so feed it the full historical+SSP range.
