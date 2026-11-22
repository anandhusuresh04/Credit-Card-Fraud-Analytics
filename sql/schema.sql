-- SQLite 3.25+ | ULB Credit Card Fraud Analytics | original rows retained.
-- Transaction amounts are stored as INTEGER cents to avoid monetary SUM drift.
PRAGMA foreign_keys = ON;
CREATE TABLE dataset_metadata (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL
);
CREATE TABLE dim_amount_band (
    band_id INTEGER PRIMARY KEY CHECK(band_id >= 0),
    lower_cents REAL NOT NULL,
    upper_cents REAL NOT NULL,
    label TEXT NOT NULL UNIQUE
);
CREATE TABLE dim_time_interval (
    interval_id INTEGER PRIMARY KEY CHECK(interval_id >= 0),
    start_elapsed_seconds INTEGER NOT NULL,
    end_elapsed_seconds INTEGER NOT NULL,
    label TEXT NOT NULL UNIQUE
);
CREATE TABLE transactions (
    original_row INTEGER PRIMARY KEY CHECK(original_row >= 0),
    time_elapsed_seconds REAL NOT NULL CHECK(time_elapsed_seconds >= 0),
    amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
    class INTEGER NOT NULL CHECK(class IN (0,1)),
    is_exact_duplicate INTEGER NOT NULL CHECK(is_exact_duplicate IN (0,1)),
    amount_band INTEGER REFERENCES dim_amount_band(band_id),
    time_interval INTEGER NOT NULL REFERENCES dim_time_interval(interval_id),
    V1 REAL NOT NULL, V2 REAL NOT NULL, V3 REAL NOT NULL, V4 REAL NOT NULL,
    V5 REAL NOT NULL, V6 REAL NOT NULL, V7 REAL NOT NULL, V8 REAL NOT NULL,
    V9 REAL NOT NULL, V10 REAL NOT NULL, V11 REAL NOT NULL, V12 REAL NOT NULL,
    V13 REAL NOT NULL, V14 REAL NOT NULL, V15 REAL NOT NULL, V16 REAL NOT NULL,
    V17 REAL NOT NULL, V18 REAL NOT NULL, V19 REAL NOT NULL, V20 REAL NOT NULL,
    V21 REAL NOT NULL, V22 REAL NOT NULL, V23 REAL NOT NULL, V24 REAL NOT NULL,
    V25 REAL NOT NULL, V26 REAL NOT NULL, V27 REAL NOT NULL, V28 REAL NOT NULL
);
CREATE INDEX idx_transactions_class ON transactions(class);
CREATE INDEX idx_transactions_band ON transactions(amount_band);
CREATE INDEX idx_transactions_time ON transactions(time_interval);
CREATE INDEX idx_transactions_duplicate ON transactions(is_exact_duplicate);
CREATE VIEW v_transactions AS
SELECT original_row, time_elapsed_seconds, amount_cents / 100.0 AS amount,
       class, is_exact_duplicate, amount_band, time_interval,
       V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,
       V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28
FROM transactions;
