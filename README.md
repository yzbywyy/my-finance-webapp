	Problem is as follows:
Financial time series data usually exhibit complex features, such as non-stationarity and "volatility clustering" (where large fluctuations are followed by large fluctuations, so do small ones). 
According to our ADF Stationarity Test (p-value: 0.737, ADF Statistic: -1.0429), the original price series is highly non-stationary. Traditional static models fail to capture these complex dynamics.
	Approach:
Because our ADF test indicated a non-stationary series, the differencing component (d=1) is set up to transform the data into a stationary series.
While ARIMA captures the mean, it fundamentally assumes constant variance (homoscedasticity) and cannot explain the heteroscedasticity common in financial data. To address this, we introduce the GARCH(1,1) model to capture the persistence of volatility over time. Furthermore, we incorporate the EGARCH(1,1) model to account for the "leverage effect"—overcoming the limitation of symmetric responses to positive and negative market shocks, effectively measuring the market's different reactions to good and bad news.
	Performance:
The test set results demonstrated that volatility-adjusted models drastically outperformed standard time-series models.The GARCH(1,1) Volatility Model achieved the best overall performance with the lowest errors across the board (RMSE: 224.88, MAE: 153.04, MAPE: 3.58%).
In contrast, the standalone ARIMA(5,1,0) model yielded significantly higher errors (RMSE: 397.77, MAPE: 7.91%), proving that ignoring volatility clustering limits predictive accuracy.
Between the volatility models, GARCH(1,1) proved to be more parsimonious and effective for this specific index futures dataset, yielding a lower AIC (2168.53) and BIC (2186.89) compared to the EGARCH(1,1) model (AIC: 2191.72, BIC: 2210.07).
