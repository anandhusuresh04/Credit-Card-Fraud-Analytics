"""Reproducible, read-only-source SQLite analytics for the ULB creditcard.csv.

Run from the project root:
    python -m src.sql_pipeline
    python -m src.sql_pipeline --raw data/raw/creditcard.csv --database data/processed/fraud_analytics.sqlite

Uses only Python standard library; requires SQLite 3.25+ for window functions.
Never mixes raw population SQL summaries with deduplicated ML holdout scores.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from contextlib import closing
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import tempfile

FEATURES = [f'V{i}' for i in range(1, 29)]
EXPECTED = ['Time', *FEATURES, 'Amount', 'Class']
ULB_SHA256 = '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
ULB_COUNTS = (284807, 492, 1081, 1825, 6012797)
QUERY_FILES = ('fraud_analysis.sql', 'business_queries.sql')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def money_cents(value: str, line_number: int) -> int:
    try:
        dec = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f'CSV line {line_number}: invalid Amount {value!r}') from exc
    if not dec.is_finite() or dec < 0 or dec != dec.quantize(Decimal('0.01')):
        raise ValueError(f'CSV line {line_number}: Amount must be finite, nonnegative, <=2 decimal places')
    cents = int(dec * 100)
    if cents > 9_000_000_000_000_000:
        raise ValueError(f'CSV line {line_number}: Amount exceeds safe integer aggregate range')
    return cents


def parse_record(row: list[str], row_id: int) -> tuple[float, tuple[float, ...], int, int]:
    if len(row) != 31:
        raise ValueError(f'CSV line {row_id+2}: expected 31 columns, got {len(row)}')
    try:
        time = float(row[0])
        feats = tuple(float(x) for x in row[1:29])
    except ValueError as exc:
        raise ValueError(f'CSV line {row_id+2}: nonnumeric Time/PCA feature') from exc
    if not math.isfinite(time) or time < 0 or any(not math.isfinite(v) for v in feats):
        raise ValueError(f'CSV line {row_id+2}: negative Time or non-finite value')
    cents = money_cents(row[29], row_id+2)
    if row[30] not in ('0', '1'):
        raise ValueError(f'CSV line {row_id+2}: Class must be 0 or 1')
    return time, feats, cents, int(row[30])


def load_source(path: Path) -> tuple[list[tuple], list[int], tuple[int, int, int, int, int]]:
    """Validate whole CSV before opening/replacing any destination database."""
    records = []
    cents_values = []
    seen = set()
    fraud = duplicates = zero = fraud_cents = 0
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != EXPECTED:
            raise ValueError(f'Unexpected header: expected {EXPECTED!r}, got {header!r}')
        for row_id, row in enumerate(reader):
            time, features, cents, klass = parse_record(row, row_id)
            # Fingerprint numeric values, not formatting or row order.
            fingerprint = hashlib.sha256(struct.pack('<30dB', time, *features, cents / 100.0, klass)).digest()
            is_duplicate = int(fingerprint in seen)
            seen.add(fingerprint)
            duplicates += is_duplicate
            fraud += klass
            zero += cents == 0
            fraud_cents += klass * cents
            cents_values.append(cents)
            records.append((row_id, time, cents, klass, is_duplicate, *features))
    if not records:
        raise ValueError('Input CSV is empty')
    return records, cents_values, (len(records), fraud, duplicates, zero, fraud_cents)


def quantile_edges(values: list[int], count: int = 10) -> list[float]:
    """Match pandas qcut linear quantiles (including duplicate-edge behavior)."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    edges = []
    for i in range(count + 1):
        pos = (n - 1) * i / count
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        edge = sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
        if not edges or edge != edges[-1]:
            edges.append(edge)
    if len(edges) < 3:
        raise ValueError('Insufficient unique Amount quantile boundaries; cannot form amount bands')
    return edges


def split_named_queries(sql_text: str) -> dict[str, str]:
    """Require one SELECT/WITH query per named block; no write statements permitted."""
    result = {}
    matches = list(re.finditer(r'^-- name: ([a-z][a-z0-9_]*)\s*$', sql_text, flags=re.M))
    if not matches:
        raise ValueError('No named SQL statements present')
    if sql_text[:matches[0].start()].strip() and not all(line.strip().startswith('--') or not line.strip() for line in sql_text[:matches[0].start()].splitlines()):
        raise ValueError('Unexpected SQL before first named query')
    for i, match in enumerate(matches):
        name = match.group(1)
        text = sql_text[match.end():matches[i+1].start() if i+1 < len(matches) else None].strip()
        if name in result or not text:
            raise ValueError(f'Duplicate/empty SQL query {name}')
        without_comments = '\n'.join(line for line in text.splitlines() if not line.strip().startswith('--')).strip()
        if not re.match(r'^(SELECT|WITH)\b', without_comments, re.I):
            raise ValueError(f'Only SELECT or WITH queries allowed: {name}')
        result[name] = text
    return result


def scalar(db: sqlite3.Connection, sql: str):
    return db.execute(sql).fetchone()[0]


def validate_database(db: sqlite3.Connection, source_counts: tuple[int, int, int, int, int], edges: list[float]):
    got = db.execute('''SELECT COUNT(*), COALESCE(SUM(class),0),
        COALESCE(SUM(is_exact_duplicate),0),
        COALESCE(SUM(CASE WHEN amount_cents=0 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END),0)
        FROM transactions''').fetchone()
    if tuple(got) != tuple(source_counts):
        raise AssertionError(f'Import reconciliation failed: SQL={got} CSV={source_counts}')
    for dimension in ('amount_band','time_interval'):
        if scalar(db, f'SELECT COUNT(*) FROM transactions WHERE {dimension} IS NULL'):
            raise AssertionError(f'Missing {dimension} mappings')
    band_totals = db.execute('''SELECT amount_band, COUNT(*), SUM(class),
         SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END)
         FROM transactions GROUP BY amount_band ORDER BY amount_band''').fetchall()
    if sum(x[1] for x in band_totals) != source_counts[0] or len(band_totals) != len(edges)-1:
        raise AssertionError('Amount quantile band counts do not reconcile')
    if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise AssertionError('Foreign key consistency failure')
    return band_totals


def build_database(records: list[tuple], cents_values: list[int], source_counts: tuple[int,int,int,int,int],
                   db_path: Path, schema_path: Path, digest: str) -> list[tuple]:
    edges = quantile_edges(cents_values)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + '.building')
    if tmp.exists():
        tmp.unlink()
    try:
        # sqlite3.Connection context managers commit/rollback but DO NOT close.
        # Windows refuses os.replace/unlink while SQLite still holds the file.
        with closing(sqlite3.connect(tmp)) as db:
            with db:
                db.execute('PRAGMA foreign_keys = ON')
                db.execute('PRAGMA journal_mode = DELETE')
                db.executescript(schema_path.read_text(encoding='utf-8'))
                db.executemany('INSERT INTO dim_amount_band VALUES (?,?,?,?)',
                               [(i, edges[i], edges[i+1], f'Q{i+1:02d}') for i in range(len(edges)-1)])
                max_interval = max(int(r[1] // 14400) for r in records)
                db.executemany('INSERT INTO dim_time_interval VALUES (?,?,?,?)',
                               [(i, i*14400, (i+1)*14400, f'{i*4:02d}–{(i+1)*4:02d} elapsed hours') for i in range(max_interval+1)])
                columns = ('original_row,time_elapsed_seconds,amount_cents,class,is_exact_duplicate,amount_band,time_interval,'
                           + ','.join(FEATURES))
                marks = ','.join('?' for _ in range(7+28))
                insert = f'INSERT INTO transactions ({columns}) VALUES ({marks})'
                buffer = []
                for r in records:
                    rowid, time, cents, klass, dup, *features = r
                    band = bisect.bisect_left(edges[1:-1], cents)
                    buffer.append((rowid, time, cents, klass, dup, band, int(time // 14400), *features))
                    if len(buffer) >= 3000:
                        db.executemany(insert, buffer)
                        buffer.clear()
                if buffer:
                    db.executemany(insert, buffer)
                metadata = {
                    'raw_csv_sha256': digest, 'population': 'raw_all_rows_including_exact_duplicates',
                    'source_amount_units': 'unspecified in ULB dataset',
                    'amount_storage': 'integer_cents',
                    'amount_bands': 'pandas qcut-compatible linear quantile edges; right-closed',
                    'time_origin': 'elapsed_seconds_from_dataset_start_NOT_clock_time',
                    'time_segment_seconds': '14400',
                    'model_predictions_included': 'false',
                    'source_rows': str(source_counts[0]),
                    'sqlite_version': sqlite3.sqlite_version,
                }
                db.executemany('INSERT INTO dataset_metadata VALUES (?,?)', metadata.items())
                band_totals = validate_database(db, source_counts, edges)
                db.commit()
        os.replace(tmp, db_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return band_totals


def export_queries(db_path: Path, sql_dir: Path, output_dir: Path, checks: tuple[int,int,int,int,int]):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sql_export_', dir=output_dir.parent) as tmp_name:
        temp = Path(tmp_name)
        # Close the read-only connection before the TemporaryDirectory is cleaned.
        with closing(sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True)) as db:
            db.execute('PRAGMA query_only = ON')
            query_names = []
            for filename in QUERY_FILES:
                queries = split_named_queries((sql_dir / filename).read_text(encoding='utf-8'))
                for name, sql in queries.items():
                    if name in query_names:
                        raise ValueError(f'Repeated query name: {name}')
                    cursor = db.execute(sql)
                    rows = cursor.fetchall()
                    with (temp / (name+'.csv')).open('w', encoding='utf-8', newline='') as out:
                        writer = csv.writer(out)
                        writer.writerow([col[0] for col in cursor.description])
                        writer.writerows(rows)
                    query_names.append(name)
            kpi = db.execute('''SELECT COUNT(*), SUM(class),SUM(is_exact_duplicate),
                        SUM(CASE WHEN amount_cents=0 THEN 1 ELSE 0 END),
                        SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) FROM transactions''').fetchone()
            if tuple(kpi) != checks:
                raise AssertionError('Post-export database counts mismatch')
            manifest = {'queries': query_names, 'source_rows': checks[0],
                        'fraud_cases': checks[1], 'duplicate_rows': checks[2],
                        'zero_amount_rows': checks[3], 'fraud_amount_source_units': checks[4]/100,
                        'currency_assumed': False,
                        'interpretation': 'Retrospective original raw population. This SQL analysis does not include ML holdout predictions.'}
            (temp / 'sql_audit.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in temp.iterdir():
            os.replace(file, output_dir / file.name)
    return query_names


def run(raw: Path, database: Path, reports: Path, sql_dir: Path):
    if sqlite3.sqlite_version_info < (3,25):
        raise RuntimeError('SQLite 3.25+ is required for window functions')
    if not raw.is_file():
        raise FileNotFoundError(f'Missing raw data: {raw}')
    for filename in ('schema.sql', *QUERY_FILES):
        if not (sql_dir / filename).is_file():
            raise FileNotFoundError(f'Missing {filename} in {sql_dir}')
    if raw.resolve() == database.resolve():
        raise ValueError('Database path cannot overwrite raw CSV')
    digest = sha256_file(raw)
    records, amounts, counts = load_source(raw)
    if digest == ULB_SHA256 and counts != ULB_COUNTS:
        raise AssertionError('Canonical ULB dataset counts differ unexpectedly')
    bands = build_database(records, amounts, counts, database, sql_dir/'schema.sql', digest)
    names = export_queries(database, sql_dir, reports, counts)
    print(f'[sql] PASS — {counts[0]:,} rows | {counts[1]} fraud | {counts[2]:,} exact duplicates | '
          f'{counts[4]/100:,.2f} fraud amount units')
    print(f'[sql] Amount bands: {len(bands)}; lowest {bands[0][1]:,} rows / {bands[0][2]} fraud; '
          f'highest {bands[-1][1]:,} rows / {bands[-1][2]} fraud')
    print(f'[sql] SQLite database -> {database}')
    print(f'[sql] Exported {len(names)} named SQL queries + audit JSON -> {reports}')
    return counts, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw/creditcard.csv'))
    parser.add_argument('--database', type=Path, default=Path('data/processed/fraud_analytics.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('reports/tables/sql'))
    parser.add_argument('--sql-dir', type=Path, default=Path('sql'))
    args = parser.parse_args()
    run(args.raw, args.database, args.output, args.sql_dir)


if __name__ == '__main__':
    main()
