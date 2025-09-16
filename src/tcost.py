"""
Transaction Cost Analysis Program

This program performs transaction cost analysis using four different mathematical models
to predict trading costs based on trade size and market volatility.

Models:
- Model A: cost = β₀ + β₁ * √(trade_size)
- Model B: cost = β₀ + β₁ * trade_size
- Model C: cost = β₀ + β₁ * √(trade_size) + β₂ * volatility + β₃ * volatility * √(trade_size)
- Model D: cost = β₀ + β₁ * trade_size + β₂ * volatility + β₃ * volatility * trade_size

Features:
- Generates simulated trading data (80,000 samples) with known parameters
- Fits models using both Mean Absolute Error (MAE) and Mean Squared Error (MSE) loss functions
- Calculates bootstrap standard errors for parameter estimates
- Computes model performance metrics (MSE, MAE, Pseudo R²)
- Generates comprehensive statistical summaries and outlier analysis
- Creates visualization plots showing cost curves across volatility percentiles
- Outputs all plots to a single PDF file for analysis

The program uses scipy.optimize for parameter estimation and provides detailed
comparison tables to evaluate model performance across different loss functions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set Up Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Define Models
def model_a(X, beta0, beta1):
    size, vol = X
    return beta0 + beta1 * (size ** 0.5)

def model_b(X, beta0, beta1):
    size, vol = X
    return beta0 + beta1 * size

def model_c(X, beta0, beta1, beta2, beta3):
    size, vol = X
    return beta0 + beta1 * (size ** 0.5) + beta2 * vol + beta3 * vol * (size ** 0.5)

def model_d(X, beta0, beta1, beta2, beta3):
    size, vol = X
    return beta0 + beta1 * size + beta2 * vol + beta3 * vol * size

models = {
    'Model_a': (model_a, [1, 1]),
    'Model_b': (model_b, [1, 1]),
    'Model_c': (model_c, [1, 1, 1, 1]),
    'Model_d': (model_d, [1, 1, 1, 1])
}

# Loss Functions
def mse_loss(params, model, X, y_true):
    y_pred = model(X, *params)
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(params, model, X, y_true):
    y_pred = model(X, *params)
    return np.mean(np.abs(y_true - y_pred))

# Simulated Dataset Generation
if 1: #switch to 0 to use actual tcost.csv data
    np.random.seed(42)
    n = 80000  # Sample size
    trade_size = 0.01*np.random.uniform(1, 10, n)
    volatility = np.random.uniform(0.01, 1.0, n)
    epsilon = np.random.normal(0, 0.0001, n)
    actual_beta_c = [0.00010, -0.00095, -0.00018, 0.02797]  # estimated betas using actual data and an MAE metric
    cost = model_c([trade_size, volatility], actual_beta_c[0], actual_beta_c[1], actual_beta_c[2], actual_beta_c[3]) + epsilon
    data = pd.DataFrame({"trade_size": trade_size, "volatility": volatility, "cost": cost})
else:
    data = pd.read_csv('tcost.csv')
    data.drop(columns=['Unnamed: 0'], inplace = True)
    data = data.iloc[:80000].copy()

print(data)
print(data.info(),'\n\n')
print(data.shape,'\n\n')
print(data.columns,'\n\n')
print(data.dtypes,'\n\n')

# Print Summary Statistics for Each Column
data = data.iloc[:80000].copy()

def describe_data(data):
    return {
        "Mean": f"{np.nanmean(data):.6g}",
        "Std Dev": f"{np.nanstd(data):.6g}",
        "Skewness": f"{skew(data, nan_policy='omit'):.6g}",
        "Kurtosis": f"{kurtosis(data, nan_policy='omit'):.6g}",
        "Min": f"{np.nanpercentile(data, 0):.6g}",
        "1st Pctl": f"{np.nanpercentile(data, 1):.6g}",
        "5th Pctl": f"{np.nanpercentile(data, 5):.6g}",
        "25th Pctl": f"{np.nanpercentile(data, 25):.6g}",
        "Median": f"{np.nanpercentile(data, 50):.6g}",
        "75th Pctl": f"{np.nanpercentile(data, 75):.6g}",
        "95th Pctl": f"{np.nanpercentile(data, 95):.6g}",
        "99th Pctl": f"{np.nanpercentile(data, 99):.6g}",
        "99.9th Pctl": f"{np.nanpercentile(data, 99.9):.6g}",
        "Max": f"{np.nanpercentile(data, 100):.6g}"
    }

input_stats = pd.DataFrame()
for column in ['cost', 'trade_size', 'volatility']:
    stats = describe_data(data[column])
    stats_df = pd.DataFrame(stats, index=[column])
    input_stats = pd.concat([input_stats, stats_df])

input_stats = input_stats.T
print("Summary statistics for Training Data:")
print(input_stats)

# Examine Data for Outliers
top_15_volatility = data.sort_values(by='volatility', ascending=False).head(15)
top_15_trade_size = data.sort_values(by='trade_size', ascending=False).head(15)
top_15_abs_cost = data.reindex(data['cost'].abs().sort_values(ascending=False).index).head(15)
head_15 = data.head(15)

print("\n15 Representative Records:")
print(head_15)

print("\nTop 15 Records Sorted by Volatility:")
print(top_15_volatility)

print("\nTop 15 Records Sorted by Trade Size:")
print(top_15_trade_size)

print("\nTop 15 Records Sorted by Absolute Value of Cost:")
print(top_15_abs_cost)

# Bootstrap Standard Errors
def bootstrap_standard_errors(model_func, loss_func, initial_params, X, y_true, n_bootstrap=20):
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        X_boot = (X[0][idx], X[1][idx])
        y_boot = y_true[idx]
        result = minimize(loss_func, initial_params, args=(model_func, X_boot, y_boot), method="L-BFGS-B")
        if result.success:
            bootstrap_estimates.append(result.x)
    return np.nanstd(np.array(bootstrap_estimates), axis=0)

# Fit Models
results = []
X = (data["trade_size"].values, data["volatility"].values)
y_true = data["cost"].values

for loss_type, loss_func in [("MAE", mae_loss), ("MSE", mse_loss)]:
    for model_name, (model_func, initial_params) in models.items():
        result = minimize(loss_func, initial_params, args=(model_func, X, y_true), method="L-BFGS-B")
        params = result.x
        mse = np.mean((y_true - model_func(X, *params)) ** 2)
        mae = np.mean(np.abs(y_true - model_func(X, *params)))
        r2 = 1 - (np.sum((y_true - model_func(X, *params))**2) / np.sum((y_true - np.mean(y_true))**2))
        std_errors = bootstrap_standard_errors(model_func, loss_func, initial_params, X, y_true)
        results.append({'Model': model_name, 'Loss': loss_type, 'MSE': mse, 'MAE': mae, 'Pseudo R²': r2,
                        'Parameters': params, 'Standard Errors': std_errors})

# Display Results
results_list = []
for row in results:
    results_list.append({
        'Model': row['Model'], 'Loss': row['Loss'], 'MSE': row['MSE'], 'MAE': row['MAE'],
        'Pseudo R²': row['Pseudo R²'], 'Parameters': [f"{param:.5f}" for param in row['Parameters']]
    })
    results_list.append({
        'Model': row['Model'], 'Loss': 'Standard Errors', 'MSE': None, 'MAE': None, 'Pseudo R²': None,
        'Parameters': [f"{se:.5f}" for se in row['Standard Errors']]
    })

results_df = pd.DataFrame([{**row, **{f"Param {i+1}": param for i, param in enumerate(row['Parameters'])}} for row in results])
print("\nFinal ModelResults:")
print(results_df)

comparison_table = results_df.pivot(index='Model', columns='Loss', values=[f'Param {i+1}' for i in range(len(models['Model_c'][1]))])

print("\nComparison Table:")
print(comparison_table)

# Illustrative Cost Plots
with PdfPages('../res/tcost_analysis_plots.pdf') as pdf:
    # MSE plots
    fig_mse, axs_mse = plt.subplots(2, 2, figsize=(14, 12))
    fig_mse.suptitle('Model Illustrative Costs (MSE Optimization)', fontsize=16)

    # MAE plots
    fig_mae, axs_mae = plt.subplots(2, 2, figsize=(14, 12))
    fig_mae.suptitle('Model Illustrative Costs (MAE Optimization)', fontsize=16)

    size_range = np.linspace(min(data["trade_size"]), max(data["trade_size"]), 100)
    vol_percentiles = np.percentile(data["volatility"], [5, 25, 50, 75, 95])

    for i, (model_name, (model_func, _)) in enumerate(models.items()):
        ax_mse = axs_mse.flat[i]
        ax_mae = axs_mae.flat[i]

        mse_params = np.mean([row['Parameters'] for row in results if row['Model'] == model_name and row['Loss'] == 'MSE'], axis=0)
        mae_params = np.mean([row['Parameters'] for row in results if row['Model'] == model_name and row['Loss'] == 'MAE'], axis=0)

        for vol in vol_percentiles:
            mse_costs = model_func((size_range, np.full_like(size_range, vol)), *mse_params)
            mae_costs = model_func((size_range, np.full_like(size_range, vol)), *mae_params)

            ax_mse.plot(size_range, mse_costs, label=f'Volatility = {vol:.2f}')
            ax_mae.plot(size_range, mae_costs, label=f'Volatility = {vol:.2f}')

        ax_mse.set_title(f"{model_name} Illustrative Costs (MSE)")
        ax_mse.set_xlabel('Trade Size')
        ax_mse.set_ylabel('Cost')
        ax_mae.set_title(f"{model_name} Illustrative Costs (MAE)")
        ax_mae.set_xlabel('Trade Size')
        ax_mae.set_ylabel('Cost')
        ax_mse.legend()
        ax_mae.legend()

    fig_mse.tight_layout()
    fig_mae.tight_layout()

    # Save both figures to PDF
    pdf.savefig(fig_mse)
    pdf.savefig(fig_mae)

    plt.close(fig_mse)
    plt.close(fig_mae)

print("\nPlots saved to ../res/tcost_analysis_plots.pdf")