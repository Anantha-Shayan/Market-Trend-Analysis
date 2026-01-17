# Probabilistic Forecasting of Agricultural Market Prices in India

## Overview
Agricultural commodity prices in India exhibit significant volatility due to seasonal effects, regional supply variations, and market dynamics. While historical daily price data is publicly available through government sources, it is rarely transformed into forward-looking, uncertainty-aware insights that can support farmer decision-making.

This project develops a **probabilistic time-series forecasting system** to model historical mandi prices and predict short-term future price trends across multiple commodities and markets. The system is designed as a **decision-support tool**, providing uncertainty-aware forecasts rather than deterministic or prescriptive recommendations.

---

## Problem Statement
Given historical daily modal prices for agricultural commodities across multiple Indian markets, the objective is to:
- Forecast short-term future prices (e.g., 7-day horizon)
- Quantify uncertainty in predictions
- Scale across many commodity–market time series using a unified model

The task is formulated as a **multi-series, univariate time-series forecasting problem**.

---

## Data Description
- **Source**: Government of India mandi price data (Agmarknet)
- **Frequency**: Daily
- **Time Span**: Multi-year (approximately 2015–2025)
- **Key Fields**:
  - Commodity
  - Market (Mandi)
  - District
  - State
  - Date
  - Modal Price (₹/quintal)

### Data Limitations
- Long-term arrival (supply) data is incomplete or unavailable for many markets
- No explicit demand indicators
- External factors such as weather, festivals, or policy changes are not included in the current phase

---

## Methodology

### Modeling Approach
- **Model**: DeepAR (Deep Autoregressive Recurrent Neural Network)
- **Architecture**: LSTM-based global model trained across all time series
- **Loss Function**: Normal Distribution Negative Log-Likelihood
- **Output**: Parameters of a probability distribution over future prices

### Training and Validation
- Sliding window encoder–decoder formulation
- Rolling-origin (time-aware) validation
- Chronologically held-out test set for final evaluation

This approach preserves temporal causality and avoids data leakage.

---

## Forecasting Output
- Probabilistic forecasts expressed as:
  - Median prediction
  - Lower and upper confidence bounds (quantiles)
- Forecasts are **advisory in nature** and explicitly communicate uncertainty
- No deterministic guarantees or action directives are generated

---

## Results Summary
- The model captures short-term price dynamics across diverse markets
- Prediction intervals widen during volatile periods, reflecting increased uncertainty
- Rolling validation demonstrates consistent performance across multiple time series

---

## Key Learnings
- Time-series problems require temporal validation strategies, not random splits
- Probabilistic forecasting is essential for risk-sensitive domains like agriculture
- Global models can leverage shared patterns across related markets
- Data quality and missing exogenous signals remain significant challenges

---

## Ethical Considerations
- The system is designed as a **decision-support tool**, not a prescriptive system
- Uncertainty bounds are explicitly communicated to avoid overconfidence
- The project avoids claims of guaranteed price movements or optimal actions

---

## Future Scope
- Integration of arrival (supply) data where available
- Inclusion of weather, seasonal, and festival-related covariates
- Drift monitoring and automated retraining mechanisms
- Extension to multivariate models (e.g., Temporal Fusion Transformer)
- Deployment as part of a farmer-facing advisory application

---

## Tech Stack
- **Language**: Python
- **Libraries**:
  - PyTorch
  - PyTorch Forecasting
  - Lightning
  - Pandas, NumPy
- **Environment**: GPU-enabled (Google Colab)

---

## Disclaimer
This project is intended for academic and research purposes. Forecasts are probabilistic estimates based on historical data and should not be interpreted as financial or trading advice.

---

## Author
*Anantha Shayan*  
Minor in AI

