# Question 1 Time-series Analysis Design

## Goal

Complete the descriptive and explanatory deliverables of Question 1 without
misrepresenting sparse proxy observations as continuously observed visitors.

## Data and interpretation

- `annual_city_tourism_1999_2024.csv` contains the official citywide annual
  domestic-tourist series for 2002--2024 (the earlier rows have a different
  inbound-tourist indicator and are excluded).
  It supports medium/long-term trend and extraordinary-shock analysis, but not
  a within-year seasonal decomposition.
- `daily_region_censored_likelihood_pressure_2023_2025.csv` is the three-region
  continuous relative-pressure estimate. It supports within-year, weekly and
  monthly patterns. It remains an estimate, not a daily official count.
- `censored_likelihood_parameter_diagnostics.csv` supplies standardized joint
  model coefficients for factor-strength ranking.
- `hourly_service_time_evidence.csv` contains service windows and qualitative
  time cues only. It can define early/daytime/evening service-demand scenarios,
  never observed hourly entry or exit proportions.

## Outputs

The analysis script will write a single, reproducible Question 1 result set:

1. city annual trend, fitted smooth trend and residual table;
2. regional monthly and weekday seasonal profiles, plus peak/shoulder/off-season
   classifications based on each region's 2023--2025 monthly median pressure;
3. an additive decomposition of each continuous regional pressure series into
   centered moving-average trend, calendar-month seasonal component and residual;
4. a factor-strength table ranked by absolute standardized joint coefficient;
5. a service-time scenario table carrying evidence source, applicable period,
   time window and an explicit `not_observed_hourly_flow` flag;
6. an auditable JSON quality report stating sample range, units and limitations.

## Methods

The city annual trend is a centered three-year moving average; the endpoint
trend uses the available one-sided window. The residual is official annual
domestic tourists minus this smooth trend. No artificial annual seasonality is
fitted.

For each region, the decomposition applies to `log1p(pressure_index)`: a
centered 31-day rolling mean is the medium-term trend, the month-of-year mean
of detrended values is the seasonal component, and the remainder is the random
component. This is deliberately transparent and reproducible; it is not called
a causal decomposition.

Monthly median pressure determines season labels within each region: values
strictly above the top quartile are peak, values strictly below the bottom
quartile are off-season, and the remainder are shoulder season. If the lower
quartile itself is zero, zero-pressure months are also labelled off-season; this
avoids leaving a clearly low estimated season unclassified merely because of a
floor tie. The label still refers to estimated relative pressure, not true zero
visitor counts.

## Error handling and tests

The analysis functions reject missing required columns, empty frames and
non-positive rolling-window sizes. Tests cover annual trend/residual identity,
regional decomposition reconstruction, season-class coverage and hourly
evidence not being converted into counts.

## Scope limits

No output asserts a real hourly visitor count, a citywide daily total or a
causal effect. Absolute daily visitor estimates remain in the separately
labelled anchor-constrained file and are not used to infer the seasonal shape.
