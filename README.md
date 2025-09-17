# Transaction Cost Analysis

A Python program for analyzing and comparing mathematical models that predict transaction costs in financial trading based on trade size and market volatility.

See pdfs in /doc for a summary of our findings. 

## Overview

This tool implements and compares four different transaction cost models:
- **Model A**: `cost = β₀ + β₁ * √(trade_size)`
- **Model B**: `cost = β₀ + β₁ * trade_size`
- **Model C**: `cost = β₀ + β₁ * √(trade_size) + β₂ * volatility + β₃ * volatility * √(trade_size)`
- **Model D**: `cost = β₀ + β₁ * trade_size + β₂ * volatility + β₃ * volatility * trade_size`

## Features

Features:
- Empirically examins 100,000 actual hedge fund trades.
- Generates 80,000 simulated trading samples with known parameters
- Fits models using MAE and MSE loss functions
- Calculates bootstrap standard errors for parameter estimates
- Computes performance metrics (MSE, MAE, Pseudo R²)
- Creates comprehensive visualizations and statistical analysis
- Outputs all plots to a single PDF file

## Usage

```bash
python src/tcost.py
```

## Output

The program generates detailed statistical summaries, model comparison tables, and visualization plots showing cost curves across different volatility percentiles.
