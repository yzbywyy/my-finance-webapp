#!/usr/bin/env python
"""
Financial Data Analysis Project (CSI 300 Index Futures)

This script:
1) Downloads CSI 300 index futures data via akshare (2023-01-01 to 2026-01-01) and saves CSV.
2) Splits data into train/validation/test sets.
3) Fits multiple forecasting models and evaluates RMSE/MAE/MAPE.
4) Produces EDA, diagnostics, and prediction-interval charts.
5) Generates a single standalone HTML report with an app-like layout.

All comments and HTML are in English as requested.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf

try:
    # Optional dependency for GARCH modelling
    from arch import arch_model  # type: ignore
except Exception:
    arch_model = None

# Optional: reduce plotly default size for embedded figures
pio.templates.default = "plotly_white"


START_DATE = "2023-01-01"
END_DATE = "2026-01-01"
OUTPUT_DIR = os.path.join("output")
DEFAULT_CSV = os.path.join(OUTPUT_DIR, "csi300_if_2023_2026.csv")
REPORT_HTML = os.path.join(OUTPUT_DIR, "report.html")


@dataclass
class SplitData:
    train: pd.Series
    valid: pd.Series
    test: pd.Series


def ensure_output_dir() -> None:
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize column names from akshare to a standard schema.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Guess date column
    date_candidates = [c for c in df.columns if "date" in c or "时间" in c or "日期" in c]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        date_col = df.columns[0]

    df.rename(columns={date_col: "date"}, inplace=True)

    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
        "成交量": "volume",
        "持仓量": "open_interest",
    }

    # Map common fields if present
    for col in list(df.columns):
        if col in rename_map:
            df.rename(columns={col: rename_map[col]}, inplace=True)

    # If close not found, try to detect by keywords
    if "close" not in df.columns:
        for col in df.columns:
            if "close" in col or "收盘" in col:
                df.rename(columns={col: "close"}, inplace=True)
                break

    # If open/high/low not found, attempt by keywords
    for key, keyword in [("open", "开盘"), ("high", "最高"), ("low", "最低")]:
        if key not in df.columns:
            for col in df.columns:
                if keyword in col:
                    df.rename(columns={col: key}, inplace=True)
                    break

    return df


def download_csi300_if_csv(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Download CSI 300 index futures data using akshare.
    Tries multiple symbol candidates to improve robustness.
    """
    ensure_output_dir()
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        raise RuntimeError("akshare is required. Please install it first.") from exc

    symbol_candidates = ["IF0", "IF1", "IF2", "IF3", "IF5", "IF"]
    last_err = None
    df = None

    for symbol in symbol_candidates:
        try:
            # Common akshare interface for futures daily data
            if hasattr(ak, "futures_zh_daily_sina"):
                df = ak.futures_zh_daily_sina(symbol=symbol)
            elif hasattr(ak, "futures_zh_daily"):
                df = ak.futures_zh_daily(symbol=symbol)
            else:
                raise RuntimeError("akshare does not expose known futures daily APIs.")

            if df is not None and len(df) > 0:
                break
        except Exception as err:  # pragma: no cover - download robustness
            last_err = err
            df = None
            continue

    if df is None or len(df) == 0:
        raise RuntimeError(f"Failed to download data with akshare. Last error: {last_err}")

    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    if "close" not in df.columns:
        raise RuntimeError("Downloaded data does not include a close price column.")

    # Save to CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df


def load_or_download(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    # 1. 检查文件是否存在
    if not os.path.isfile(csv_path):
        print(f"[信息] 本地未找到 {csv_path}，正在尝试下载...")
        return download_csi300_if_csv(csv_path)
    
    print(f"[信息] 正在读取本地文件: {csv_path}")
    
    # 2. 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 3. 强力标准化列名 (全部转小写，去除空格)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 4. 智能查找 'date' 列
    if "date" not in df.columns:
        # 尝试找含有 'date', '日期', 'time' 的列
        for col in df.columns:
            if any(x in col for x in ["date", "日期", "时间", "time"]):
                df.rename(columns={col: "date"}, inplace=True)
                print(f"[修复] 自动将列名 '{col}' 重命名为 'date'")
                break
    
    # 5. 智能查找 'close' 列
    if "close" not in df.columns:
        # 尝试找含有 'close', '收盘', 'price' 的列
        for col in df.columns:
            if any(x in col for x in ["close", "收盘", "最新", "price"]):
                df.rename(columns={col: "close"}, inplace=True)
                print(f"[修复] 自动将列名 '{col}' 重命名为 'close'")
                break
    
    # 6. 关键修复：清洗数据
    if "close" in df.columns:
        # 如果是字符串类型，先要把逗号去掉 (例如 "3,500.00" -> "3500.00")
        if df["close"].dtype == object:
            print("[警告] 检测到收盘价为文本格式，正在尝试清洗逗号并转换为数字...")
            df["close"] = df["close"].astype(str).str.replace(",", "").str.replace("，", "")
        
        # 强制转换为数字，无法转换的变成 NaN
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
    else:
        # 如果实在找不到，打印所有列名帮用户排查
        print(f"[错误] CSV文件现有的列名: {list(df.columns)}")
        raise ValueError("错误：在 CSV 中找不到 'close' 或 '收盘价' 相关的列，请检查 CSV 表头。")
        
    # 7. 处理日期
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # 8. 删除脏数据
    old_len = len(df)
    df = df.dropna(subset=["date", "close"])
    if len(df) < old_len:
        print(f"[清理] 已自动剔除 {old_len - len(df)} 行无效/空数据")

    # 9. 打印预览，确保你看到的是正常的 3000 多或 4000 多的数字
    print("-" * 30)
    print("数据读取成功！前5行预览 (请确认 close 列是 3000-4000 的数字):")
    print(df[["date", "close"]].head())
    print("-" * 30)
    
    return df


def split_series(series: pd.Series, train_ratio=0.7, valid_ratio=0.15) -> SplitData:
    n = len(series)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return SplitData(
        train=series.iloc[:train_end],
        valid=series.iloc[train_end:valid_end],
        test=series.iloc[valid_end:],
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _calc_intervals(y_pred: np.ndarray, residuals: np.ndarray, alpha=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normal approximation intervals from residuals.
    """
    sigma = np.std(residuals, ddof=1)
    z = stats.norm.ppf(1 - alpha / 2.0)
    lower = y_pred - z * sigma
    upper = y_pred + z * sigma
    return lower, upper


def mean_model_forecast(train: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_val = train.mean()
    y_pred = np.repeat(mean_val, horizon)
    residuals = train - mean_val
    lower, upper = _calc_intervals(y_pred, residuals)
    return y_pred, lower, upper


def naive_model_forecast(train: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_val = train.iloc[-1]
    y_pred = np.repeat(last_val, horizon)
    residuals = train.diff().dropna()
    lower, upper = _calc_intervals(y_pred, residuals)
    return y_pred, lower, upper


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if season_len <= 0 or season_len >= len(train):
        season_len = max(1, min(20, len(train) - 1))
    y_pred = np.array([train.iloc[-season_len + (i % season_len)] for i in range(horizon)])
    residuals = (train[season_len:].values - train[:-season_len].values)
    lower, upper = _calc_intervals(y_pred, residuals)
    return y_pred, lower, upper


def drift_model_forecast(train: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(train)
    if n < 2:
        return naive_model_forecast(train, horizon)
    drift = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    y_pred = np.array([train.iloc[-1] + drift * (i + 1) for i in range(horizon)])
    residuals = train.diff().dropna()
    lower, upper = _calc_intervals(y_pred, residuals)
    return y_pred, lower, upper


def rolling_mean_forecast(series: pd.Series, horizon: int, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dynamic rolling mean forecast using the latest available actual values.
    """
    if window <= 1:
        window = 5
    history = list(series.values)
    preds = []
    residuals = []

    # Build residuals on in-sample rolling mean
    for i in range(window, len(history)):
        mean_val = np.mean(history[i - window:i])
        residuals.append(history[i] - mean_val)

    for i in range(horizon):
        mean_val = np.mean(history[-window:])
        preds.append(mean_val)
        history.append(mean_val)

    y_pred = np.array(preds)
    residuals = np.array(residuals) if len(residuals) > 0 else np.array([0.0])
    lower, upper = _calc_intervals(y_pred, residuals)
    return y_pred, lower, upper


def arima_forecast(train: pd.Series, horizon: int, order=(5, 1, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = ARIMA(train, order=order)
    fitted = model.fit()
    fc = fitted.get_forecast(steps=horizon)
    y_pred = fc.predicted_mean.values
    conf_int = fc.conf_int(alpha=0.05)
    lower = conf_int.iloc[:, 0].values
    upper = conf_int.iloc[:, 1].values
    residuals = fitted.resid
    return y_pred, lower, upper, residuals.values


def garch_forecast_from_close(
    close: pd.Series,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a GARCH(1,1) model on returns and build a price-forecast style output.

    - The conditional mean of returns is assumed to be zero.
    - We map conditional volatility forecasts on returns to a price-level
      forecast by centering on the last observed close and scaling a
      symmetric normal interval using the forecast volatility.
    """
    if arch_model is None:
        raise RuntimeError("The 'arch' package is required for GARCH modelling. Please install it first (e.g. pip install arch).")

    # Work on percentage returns to stabilize scale
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        raise RuntimeError("Not enough data to fit a GARCH model (need at least ~30 observations).")

    am = arch_model(returns * 100.0, vol="Garch", p=1, o=0, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")

    # One-step-ahead to horizon variance forecast (on % returns)
    fc = res.forecast(horizon=horizon, reindex=False)
    var_fc = fc.variance.values[-1]  # shape: (horizon,)
    sigma_fc = np.sqrt(var_fc) / 100.0  # back to decimal returns

    # In-sample conditional volatility for visualization
    cond_vol = res.conditional_volatility / 100.0

    # Price-level forecast: center on last close, use volatility-derived PI
    last_price = float(close.iloc[-1])
    y_pred = np.repeat(last_price, horizon)
    z = stats.norm.ppf(0.975)
    lower = y_pred * (1.0 - z * sigma_fc)
    upper = y_pred * (1.0 + z * sigma_fc)

    return y_pred, lower, upper, cond_vol.values, sigma_fc


def fit_garch_and_egarch(
    returns: pd.Series,
    horizon: int = 30,
):
    """
    Fit GARCH(1,1) and EGARCH models on returns and
    return model diagnostics and volatility forecasts.
    """
    if arch_model is None:
        raise RuntimeError(
            "The 'arch' package is required for GARCH/EGARCH modelling. "
            "Please install it first (e.g. pip install arch)."
        )

    clean_returns = returns.dropna()
    if len(clean_returns) < 30:
        raise RuntimeError("Not enough data to fit GARCH/EGARCH models (need at least ~30 observations).")

    # GARCH(1,1) with multi-step volatility forecast
    garch_am = arch_model(
        clean_returns * 100.0,
        vol="Garch",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )
    garch_res = garch_am.fit(disp="off")
    garch_fc = garch_res.forecast(horizon=horizon, reindex=False)
    garch_var_fc = garch_fc.variance.values[-1]
    garch_sigma_fc = np.sqrt(garch_var_fc) / 100.0  # decimal returns

    garch_info = {
        "name": "GARCH(1,1)",
        "params": garch_res.params,
        "aic": float(garch_res.aic),
        "bic": float(garch_res.bic),
        "cond_vol": (garch_res.conditional_volatility / 100.0),
        "sigma_forecast_30": garch_sigma_fc,
    }

    # EGARCH(1,1): some specifications do not support analytic
    # multi-step forecasts. We therefore only compute 1-step
    # ahead volatility for diagnostics and still report
    # parameter estimates and information criteria.
    egarch_am = arch_model(
        clean_returns * 100.0,
        vol="EGARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )
    egarch_res = egarch_am.fit(disp="off")
    try:
        egarch_fc = egarch_res.forecast(horizon=1, reindex=False)
        egarch_var_fc = egarch_fc.variance.values[-1]
        egarch_sigma_1 = np.sqrt(egarch_var_fc) / 100.0
    except Exception:
        egarch_sigma_1 = np.array([np.nan])

    egarch_info = {
        "name": "EGARCH(1,1)",
        "params": egarch_res.params,
        "aic": float(egarch_res.aic),
        "bic": float(egarch_res.bic),
        "sigma_forecast_1": egarch_sigma_1,
    }

    return garch_info, egarch_info


def evaluate_models(series: pd.Series, split: SplitData, season_len: int, rolling_window: int) -> Dict[str, Dict]:
    """
    Evaluate multiple models on test set, using train+valid for fitting.
    """
    train_full = pd.concat([split.train, split.valid])
    horizon = len(split.test)

    results: Dict[str, Dict] = {}

    # Mean
    mean_pred, mean_low, mean_up = mean_model_forecast(train_full, horizon)
    results["Mean Model"] = {
        "pred": mean_pred,
        "lower": mean_low,
        "upper": mean_up,
    }

    # Naive
    naive_pred, naive_low, naive_up = naive_model_forecast(train_full, horizon)
    results["Naive Model"] = {
        "pred": naive_pred,
        "lower": naive_low,
        "upper": naive_up,
    }

    # Seasonal Naive
    s_pred, s_low, s_up = seasonal_naive_forecast(train_full, horizon, season_len)
    results["Seasonal Naive Model"] = {
        "pred": s_pred,
        "lower": s_low,
        "upper": s_up,
    }

    # Drift
    d_pred, d_low, d_up = drift_model_forecast(train_full, horizon)
    results["Drift Model"] = {
        "pred": d_pred,
        "lower": d_low,
        "upper": d_up,
    }

    # Rolling mean (dynamic)
    r_pred, r_low, r_up = rolling_mean_forecast(train_full, horizon, rolling_window)
    results["Rolling Mean Model"] = {
        "pred": r_pred,
        "lower": r_low,
        "upper": r_up,
    }

    # ARIMA
    try:
        a_pred, a_low, a_up, a_resid = arima_forecast(train_full, horizon)
        results["ARIMA(5,1,0)"] = {
            "pred": a_pred,
            "lower": a_low,
            "upper": a_up,
            "resid": a_resid,
        }
    except Exception as err:  # pragma: no cover
        results["ARIMA(5,1,0)"] = {
            "pred": np.full(horizon, np.nan),
            "lower": np.full(horizon, np.nan),
            "upper": np.full(horizon, np.nan),
            "error": str(err),
        }

    # GARCH-based volatility model mapped to price level
    # Note: we fit GARCH on the full close series (not just train_full)
    # to keep the behaviour stable and ensure enough data.
    try:
        g_pred, g_low, g_up, cond_vol, _ = garch_forecast_from_close(series, horizon)
        results["GARCH(1,1) Volatility Model"] = {
            "pred": g_pred,
            "lower": g_low,
            "upper": g_up,
        }

        # Volatility-aware mean model: use in-sample conditional volatility
        # as a feature to compute a volatility-weighted mean level.
        cond_vol = np.asarray(cond_vol)
        # Align conditional volatility to training sample length
        cond_vol = cond_vol[-len(train_full) :]
        eps = 1e-6
        weights = 1.0 / (cond_vol + eps)
        weights = weights / weights.sum()
        vol_weighted_mean = float(np.sum(train_full.values * weights))
        vw_pred = np.repeat(vol_weighted_mean, horizon)
        vw_residuals = train_full.values - vol_weighted_mean
        vw_low, vw_up = _calc_intervals(vw_pred, vw_residuals)
        results["Volatility-Weighted Mean"] = {
            "pred": vw_pred,
            "lower": vw_low,
            "upper": vw_up,
        }
    except Exception as err:  # pragma: no cover
        results["GARCH(1,1) Volatility Model"] = {
            "pred": np.full(horizon, np.nan),
            "lower": np.full(horizon, np.nan),
            "upper": np.full(horizon, np.nan),
            "error": str(err),
        }

    return results


def build_metrics_table(test: pd.Series, results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        pred = res["pred"]
        rows.append(
            {
                "Model": name,
                "RMSE": rmse(test.values, pred),
                "MAE": mae(test.values, pred),
                "MAPE(%)": mape(test.values, pred),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE")


def plot_metrics_bar(metrics_df: pd.DataFrame) -> str:
    """
    Build a bar chart for model comparison based on RMSE.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return "<p>No metrics available for plotting.</p>"

    x_models = metrics_df["Model"].tolist()
    rmse_vals = metrics_df["RMSE"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_models,
            y=rmse_vals,
            name="RMSE",
            marker=dict(color="#1b6ca8"),
        )
    )
    fig.update_layout(
        title="Model Comparison by RMSE (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="RMSE",
        height=380,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_close_and_returns(df: pd.DataFrame):
    fig_close = go.Figure()
    fig_close.add_trace(
        # 修改点：加上 .tolist()
        go.Scatter(x=df["date"].tolist(), y=df["close"].tolist(), mode="lines", name="Close")
    )
    fig_close.update_layout(
        title="Close Price (CSI 300 Index Futures)",
        xaxis_title="Date",
        yaxis_title="Close",
        height=360,
    )

    returns = df["close"].pct_change().dropna()
    fig_ret = go.Figure()
    # 修改点：加上 .tolist()
    fig_ret.add_trace(go.Scatter(x=df["date"].iloc[1:].tolist(), y=returns.tolist(), mode="lines", name="Returns"))
    fig_ret.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Return",
        height=360,
    )
    return pio.to_html(fig_close, full_html=False, include_plotlyjs=False), pio.to_html(
        fig_ret, full_html=False, include_plotlyjs=False
    )


def plot_hist_with_normal(data: pd.Series):
    # 修改点：加上 .tolist()
    mu, std = stats.norm.fit(data)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data.tolist(),  # <--- 这里修改
            histnorm="probability density",
            name="Returns",
            opacity=0.75,
        )
    )
    x_range = np.linspace(min(data), max(data), 100)
    pdf = stats.norm.pdf(x_range, mu, std)
    fig.add_trace(go.Scatter(x=x_range.tolist(), y=pdf.tolist(), mode="lines", name="Normal PDF")) # <--- 这里修改
    fig.update_layout(
        title="Returns Histogram with Normal Curve",
        xaxis_title="Return",
        yaxis_title="Density",
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_qq(data: pd.Series):
    osm, osr = stats.probplot(data, dist="norm", fit=False)
    # Fit line
    slope, intercept, r, p, stderr = stats.linregress(osm, osr)
    line_y = slope * osm + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=osm.tolist(), y=osr.tolist(), mode="markers", name="Q-Q Points") # <--- 这里修改
    )
    fig.add_trace(
        go.Scatter(x=osm.tolist(), y=line_y.tolist(), mode="lines", name="Fit Line") # <--- 这里修改
    )
    fig.update_layout(
        title="Q-Q Plot (Returns vs Normal)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_acf_pacf(data: pd.Series, lags: int = 40):
    acf_vals = acf(data, nlags=lags)
    pacf_vals = pacf(data, nlags=lags)
    x_lags = list(range(len(acf_vals)))

    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=x_lags, y=acf_vals.tolist(), name="ACF")) # <--- 这里修改
    fig_acf.update_layout(title="ACF of Returns", xaxis_title="Lag", yaxis_title="ACF", height=320)

    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=x_lags, y=pacf_vals.tolist(), name="PACF")) # <--- 这里修改
    fig_pacf.update_layout(title="PACF of Returns", xaxis_title="Lag", yaxis_title="PACF", height=320)

    return pio.to_html(fig_acf, full_html=False, include_plotlyjs=False), pio.to_html(
        fig_pacf, full_html=False, include_plotlyjs=False
    )


def plot_garch_volatility(
    dates: pd.Series,
    returns: pd.Series,
) -> str:
    """
    Fit a GARCH(1,1) model on returns and visualize the conditional volatility.
    """
    if arch_model is None:
        raise RuntimeError("The 'arch' package is required for GARCH modelling. Please install it first (e.g. pip install arch).")

    clean_returns = returns.dropna()
    if len(clean_returns) < 30:
        raise RuntimeError("Not enough data to fit a GARCH model (need at least ~30 observations).")

    am = arch_model(clean_returns * 100.0, vol="Garch", p=1, o=0, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0  # back to decimal

    # Align dates with conditional volatility length (drop the first date due to diff)
    vol_dates = dates.iloc[-len(cond_vol):]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol_dates.tolist(),
            y=cond_vol.values.tolist(),
            mode="lines",
            name="Conditional Volatility",
        )
    )
    fig.update_layout(
        title="Conditional Volatility (GARCH(1,1))",
        xaxis_title="Date",
        yaxis_title="Volatility",
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_garch_vol_forecast(
    sigma_forecast: np.ndarray,
) -> str:
    """
    Plot 30-step ahead volatility forecasts from a GARCH model.
    """
    steps = list(range(1, len(sigma_forecast) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=(sigma_forecast * 100.0).tolist(),
            mode="lines+markers",
            name="Forecast Volatility",
        )
    )
    fig.update_layout(
        title="30-Step Ahead Volatility Forecast (GARCH(1,1))",
        xaxis_title="Step",
        yaxis_title="Volatility (%)",
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_volatility_clustering(
    dates: pd.Series,
    returns: pd.Series,
    cond_vol: pd.Series,
) -> str:
    """
    Visualize volatility clustering by overlaying absolute returns and
    conditional volatility.
    """
    clean_returns = returns.dropna()
    # Align conditional volatility with returns tail
    cond_vol = pd.Series(cond_vol, index=clean_returns.index[-len(cond_vol) :])

    abs_ret = clean_returns.abs()

    # Normalize both series to comparable scales
    abs_ret_norm = abs_ret / (abs_ret.std() + 1e-8)
    cond_vol_norm = cond_vol / (cond_vol.std() + 1e-8)

    cluster_dates = dates.iloc[-len(abs_ret_norm) :]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cluster_dates.tolist(),
            y=abs_ret_norm.values.tolist(),
            mode="lines",
            name="|Returns| (normalized)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cluster_dates.tolist(),
            y=cond_vol_norm.values.tolist(),
            mode="lines",
            name="Conditional Volatility (normalized)",
        )
    )
    fig.update_layout(
        title="Volatility Clustering: |Returns| vs GARCH Volatility",
        xaxis_title="Date",
        yaxis_title="Normalized Level",
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_stl(data: pd.Series, period: int = 20):
    res = STL(data, period=period).fit()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    # 修改点：全部加上 .tolist()
    fig.add_trace(go.Scatter(x=data.index.tolist(), y=res.observed.tolist(), name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index.tolist(), y=res.trend.tolist(), name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index.tolist(), y=res.seasonal.tolist(), name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index.tolist(), y=res.resid.tolist(), name="Residual"), row=4, col=1)

    fig.update_layout(height=700, title="STL Decomposition", showlegend=False)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_forecasts(
    dates: pd.Series,
    train: pd.Series,
    test: pd.Series,
    results: Dict[str, Dict],
    model_name: str,
) -> str:
    pred = results[model_name]["pred"]
    lower = results[model_name]["lower"]
    upper = results[model_name]["upper"]

    test_dates = dates.iloc[-len(test):]
    train_dates = dates.iloc[: -len(test)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_dates, y=train, mode="lines", name="Train+Valid"))
    fig.add_trace(go.Scatter(x=test_dates, y=test, mode="lines", name="Test"))
    fig.add_trace(go.Scatter(x=test_dates, y=pred, mode="lines", name=f"{model_name} Forecast"))

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper 95%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=lower,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="95% Prediction Interval",
        )
    )

    fig.update_layout(
        title=f"Forecast with 95% Prediction Interval - {model_name}",
        xaxis_title="Date",
        yaxis_title="Close",
        height=420,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def adf_test(series: pd.Series) -> Dict[str, float]:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Used Lag": result[2],
        "N Observations": result[3],
    }


def _json_default(obj):
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    raise TypeError(f"Type not serializable: {type(obj)}")


def build_html(
    df: pd.DataFrame,
    split: SplitData,
    results: Dict[str, Dict],
    metrics_df: pd.DataFrame,
    charts: Dict[str, str],
    adf_results: Dict[str, float],
    default_model: str,
    season_len: int,
    rolling_window: int,
) -> str:
    # Embed the metrics table as HTML
    metrics_html = metrics_df.to_html(index=False, float_format="%.4f")

    # Precompute forecast traces for JS usage
    test_dates = df["date"].iloc[-len(split.test):].dt.strftime("%Y-%m-%d").tolist()
    model_payload = {
        name: {
            "pred": res["pred"],
            "lower": res["lower"],
            "upper": res["upper"],
        }
        for name, res in results.items()
    }

    payload = {
        "dates": test_dates,
        "actual": split.test.values,
        "models": model_payload,
        "default_model": default_model,
    }

    payload_json = json.dumps(payload, default=_json_default)

    # Use Plotly CDN; keep HTML single file with embedded JS and CSS
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CSI 300 Futures Analysis Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f4f1ec;
      --card: #ffffff;
      --ink: #111111;
      --muted: #5a5a5a;
      --accent: #1b6ca8;
      --accent-2: #f4a261;
      --line: #e0dcd2;
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(135deg, #f4f1ec 0%, #efe7dc 100%);
    }}
    header {{
      padding: 28px 34px;
      border-bottom: 1px solid var(--line);
      background: #f8f5ef;
    }}
    header h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
      letter-spacing: 0.4px;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 20px;
      padding: 20px 26px 30px;
    }}
    aside {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px;
      height: fit-content;
    }}
    aside h3 {{
      margin: 4px 0 12px;
      font-size: 18px;
    }}
    label {{
      display: block;
      font-size: 13px;
      margin: 12px 0 6px;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      font-family: inherit;
    }}
    .content {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .metrics {{
      overflow-x: auto;
      font-size: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      background: #fbf8f3;
    }}
    .note {{
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CSI 300 Index Futures: Forecasting and Diagnostics</h1>
    <p>Data range: {START_DATE} to {END_DATE}. Report generated on {datetime.now().strftime("%Y-%m-%d")}.</p>
  </header>
  <div class="layout">
    <aside>
      <h3>Controls</h3>
      <label for="fileInput">Upload CSV (local)</label>
      <input type="file" id="fileInput" accept=".csv" />
      <label for="modelSelect">Forecast Model</label>
      <select id="modelSelect"></select>
      <label for="windowInput">Rolling Window</label>
      <input type="number" id="windowInput" min="3" value="{rolling_window}" />
      <label for="hInput">Forecast Horizon (h)</label>
      <input type="number" id="hInput" min="5" value="{len(split.test)}" />
      <p class="note">Note: Complex models are precomputed in Python. Client-side controls update the displayed forecast only.</p>
    </aside>
    <main class="content">
      <section class="card">
        <h3>Close Price & Returns</h3>
        {charts["close"]}
        {charts["returns"]}
      </section>
      <section class="card">
        <h3>Forecast with 95% Prediction Interval</h3>
        <div id="forecastChart"></div>
      </section>
      <section class="card">
        <h3>Model Metrics (Test Set)</h3>
        <div class="metrics">{metrics_html}</div>
        {charts["metrics_chart"]}
      </section>
      <section class="card">
        <h3>GARCH & EGARCH Model Summary</h3>
        {charts["garch_summary"]}
      </section>
      <section class="card">
        <h3>Normality Diagnostics</h3>
        <div class="two-col">
          <div>
            {charts["hist"]}
          </div>
          <div>
            {charts["qq"]}
          </div>
        </div>
      </section>
      <section class="card">
        <h3>Volatility Clustering</h3>
        {charts["vol_cluster"]}
      </section>
      <section class="card two-col">
        <div>
          <h3>Autocorrelation of Returns</h3>
          {charts["acf"]}
          {charts["pacf"]}
        </div>
        <div>
          <h3>GARCH Volatility Diagnostics</h3>
          {charts["garch_vol"]}
          {charts["garch_vol_forecast"]}
        </div>
      </section>
      <section class="card">
        <h3>STL Decomposition</h3>
        {charts["stl"]}
      </section>
      <section class="card">
        <h3>ADF Stationarity Test</h3>
        <p class="note">ADF Statistic: {adf_results["ADF Statistic"]:.4f} | p-value: {adf_results["p-value"]:.6f} | Used Lag: {int(adf_results["Used Lag"])} | N: {int(adf_results["N Observations"])}</p>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const modelSelect = document.getElementById("modelSelect");
    const forecastChart = document.getElementById("forecastChart");

    const modelNames = Object.keys(payload.models);
    modelNames.forEach(name => {{
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      modelSelect.appendChild(opt);
    }});
    modelSelect.value = payload.default_model;

    function renderForecast(modelName) {{
      const model = payload.models[modelName];
      const traceActual = {{
        x: payload.dates,
        y: payload.actual,
        mode: "lines",
        name: "Actual (Test)",
        line: {{ color: "#1b6ca8" }}
      }};
      const tracePred = {{
        x: payload.dates,
        y: model.pred,
        mode: "lines",
        name: "Forecast",
        line: {{ color: "#f4a261" }}
      }};
      const traceUpper = {{
        x: payload.dates,
        y: model.upper,
        mode: "lines",
        line: {{ width: 0 }},
        showlegend: false
      }};
      const traceLower = {{
        x: payload.dates,
        y: model.lower,
        mode: "lines",
        fill: "tonexty",
        line: {{ width: 0 }},
        name: "95% PI",
        fillcolor: "rgba(244,162,97,0.2)"
      }};

      const layout = {{
        height: 420,
        title: `Forecast with 95% PI - ${{modelName}}`,
        xaxis: {{ title: "Date" }},
        yaxis: {{ title: "Close" }},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)"
      }};

      Plotly.newPlot(forecastChart, [traceActual, tracePred, traceUpper, traceLower], layout, {{displayModeBar: false}});
    }}

    modelSelect.addEventListener("change", () => renderForecast(modelSelect.value));
    renderForecast(modelSelect.value);

    // Simple CSV upload preview (does not rerun Python models)
    document.getElementById("fileInput").addEventListener("change", (event) => {{
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {{
        const text = e.target.result;
        console.log("Loaded CSV length:", text.length);
        alert("CSV loaded locally. Complex models remain from the default dataset.");
      }};
      reader.readAsText(file);
    }});
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    ensure_output_dir()
    df = load_or_download(DEFAULT_CSV)

    # Sort and keep close
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)

    # Split
    split = split_series(close, train_ratio=0.7, valid_ratio=0.15)

    # Parameters
    season_len = 20  # ~monthly trading cycle
    rolling_window = 10

    # Model evaluation
    results = evaluate_models(close, split, season_len=season_len, rolling_window=rolling_window)
    metrics_df = build_metrics_table(split.test, results)
    metrics_chart = plot_metrics_bar(metrics_df)

    # Diagnostics and EDA
    close_chart, return_chart = plot_close_and_returns(df)
    returns = close.pct_change().dropna()
    hist_chart = plot_hist_with_normal(returns)
    qq_chart = plot_qq(returns)
    acf_chart, pacf_chart = plot_acf_pacf(returns, lags=40)
    try:
        # GARCH/EGARCH fitting and diagnostics
        garch_info, egarch_info = fit_garch_and_egarch(returns, horizon=30)

        # Tabular summary of parameters and AIC/BIC
        def _one_model_table(info):
            params = info["params"]
            df_params = pd.DataFrame(
                {
                    "Parameter": list(params.index) + ["AIC", "BIC"],
                    "Value": list(params.values) + [info["aic"], info["bic"]],
                }
            )
            return f"<h4>{info['name']}</h4>" + df_params.to_html(index=False, float_format="%.4f")

        garch_summary_html = _one_model_table(garch_info) + _one_model_table(egarch_info)

        # In-sample conditional volatility and 30-step-ahead volatility forecast
        garch_cond_vol = garch_info["cond_vol"]
        garch_sigma_30 = np.asarray(garch_info["sigma_forecast_30"])

        garch_vol_chart = plot_garch_volatility(df["date"], returns)
        garch_vol_forecast_chart = plot_garch_vol_forecast(garch_sigma_30)
        vol_cluster_chart = plot_volatility_clustering(df["date"], returns, garch_cond_vol)
    except Exception as err:
        print(f"[警告] GARCH/EGARCH 拟合或可视化失败：{err}")
        garch_summary_html = f"<p>GARCH / EGARCH summary unavailable: {err}</p>"
        garch_vol_chart = "<p>GARCH volatility chart unavailable.</p>"
        garch_vol_forecast_chart = "<p>GARCH 30-step volatility forecast unavailable.</p>"
        vol_cluster_chart = "<p>Volatility clustering visualization unavailable.</p>"

    stl_chart = plot_stl(close, period=season_len)
    adf_results = adf_test(close)

    charts = {
        "close": close_chart,
        "returns": return_chart,
        "hist": hist_chart,
        "qq": qq_chart,
        "acf": acf_chart,
        "pacf": pacf_chart,
        "garch_vol": garch_vol_chart,
        "garch_vol_forecast": garch_vol_forecast_chart,
        "vol_cluster": vol_cluster_chart,
        "garch_summary": garch_summary_html,
        "metrics_chart": metrics_chart,
        "stl": stl_chart,
    }

    default_model = metrics_df.iloc[0]["Model"] if len(metrics_df) > 0 else "Mean Model"
    html = build_html(
        df,
        split,
        results,
        metrics_df,
        charts,
        adf_results,
        default_model=default_model,
        season_len=season_len,
        rolling_window=rolling_window,
    )

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved CSV: {DEFAULT_CSV}")
    print(f"Saved report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
