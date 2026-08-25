# Data acquisition

Raw WRDS data is licensed and is not included in the source or result archives.

The completed CRSP reproduction used WRDS query `11596346` with:

- Product: CRSP Annual Update → Stock Version 2 (CIZ) → Daily Stock File
- Dates: 2006-01-01 through 2025-12-31
- Code type: Ticker
- Tickers:

```text
AAPL MSFT AMZN NVDA JPM BAC GS WFC XOM CVX COP JNJ MRK PFE UNH PG KO PEP WMT COST HD MCD CAT HON BA UPS IBM CSCO ORCL TXN NEE SO
```

- Variables: `PERMNO`, `Ticker`, `DlyCalDt`, `DlyCap`, `DlyRet`
- Output: comma-delimited, gzip compressed, ISO date

Save the downloaded file as:

```text
data/raw/crsp_daily_2006_2025.csv.gz
```

The loader applies the predeclared 98% return-coverage rule, collapses duplicate
distribution-event rows deterministically, and uses PERMNO as the stable asset
identifier. In the retrieved file, NEE/PERMNO 24205 did not pass the coverage
rule, leaving 31 assets and 5,031 complete dates.

## ChiNext alternative

If the WRDS institution is entitled to CSMAR Trades, the original geographic
setting can be restored with `csmar_trade.trd_dalyr`, `Markettype = 16`, and
`Dretwd` (daily return with cash dividends reinvested). The Cornell entitlement
used here did not include `csmar_trade`, so the supplied report does not claim
to use ChiNext data.

