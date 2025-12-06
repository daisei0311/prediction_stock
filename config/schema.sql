-- Stock Prices Table
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

-- Index for faster queries by date
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date);
