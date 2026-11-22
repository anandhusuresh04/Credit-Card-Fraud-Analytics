"""Run: python -m unittest discover -s tests -p test_sql_pipeline.py -v"""
import csv
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.sql_pipeline import (EXPECTED, parse_record, money_cents, quantile_edges,
                              split_named_queries, run)

SQL_DIR = Path(__file__).resolve().parents[1] / 'sql'


def fixture(path, amounts, labels, times=None):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(EXPECTED)
        for i, (amount, label) in enumerate(zip(amounts, labels)):
            writer.writerow([(times or list(range(len(amounts))))[i], *([float(i%4)]*28), amount, label])


class SqlPipelineTests(unittest.TestCase):
    def test_money_validation(self):
        self.assertEqual(money_cents('149.62',2),14962)
        for value in ('-1', 'nan', 'inf', '0.001', 'abc'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                money_cents(value,2)

    def test_input_validation(self):
        row = ['0'] + ['1']*28 + ['10.00','1']
        self.assertEqual(parse_record(row,0)[-1],1)
        for index, bad in [(0, '-1'), (1, 'nan'), (30,'3')]:
            changed = row.copy()
            changed[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_record(changed,0)

    def test_quantile_ties_and_constant_rejected(self):
        edges = quantile_edges([0]*20 + list(range(1,81)))
        self.assertEqual(edges[0],0)
        self.assertEqual(edges[-1],80)
        with self.assertRaises(ValueError):
            quantile_edges([5]*100)

    def test_query_blocks_readonly(self):
        a = split_named_queries('-- name: first\nSELECT 1;\n-- name: second\nWITH x AS (SELECT 1) SELECT * FROM x;')
        self.assertEqual(list(a), ['first','second'])
        with self.assertRaises(ValueError):
            split_named_queries('-- name: bad\nDELETE FROM transactions;')

    def test_full_small_fixture_reconciles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root/'creditcard.csv'
            # 20 values, 10 bands, 1 exact duplicated record (same features/time/class).
            fixture(raw, [f'{v}.00' for v in range(20)], [1 if v in (0,19) else 0 for v in range(20)])
            db = root/'demo.sqlite'
            output = root/'out'
            counts, names = run(raw,db,output,SQL_DIR)
            self.assertEqual(counts,(20,2,0,1,1900))
            self.assertIn('amount_band_exposure',names)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0],20)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM dim_amount_band').fetchone()[0],10)
                self.assertEqual(conn.execute('SELECT SUM(amount_cents) FROM transactions WHERE class=1').fetchone()[0],1900)
            with (output/'executive_kpis.csv').open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual(int(row['fraud_cases']),2)
            self.assertEqual(float(row['fraudulent_value']),19.0)

    def test_input_invalid_does_not_overwrite_db(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'raw.csv'; db=root/'existing.sqlite'
            db.write_bytes(b'original remains')
            fixture(raw,['0.00']*12,[0]*12)
            with self.assertRaises(ValueError):
                run(raw,db,root/'out',SQL_DIR)
            self.assertEqual(db.read_bytes(),b'original remains')

    def test_duplicate_records_and_time_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'creditcard.csv'; db=root/'db.sqlite'
            # Row 1 exactly repeats row 0 (same time, features, amount, class).
            amounts = [f'{i}.00' for i in range(20)]
            labels = [1 if i in (0,19) else 0 for i in range(20)]
            times = [0, *list(range(1,20))]
            fixture(raw,amounts,labels,times)
            with raw.open('a', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerow([0,*([0.0]*28),'0.00',1])
            counts, _ = run(raw,db,root/'out',SQL_DIR)
            self.assertEqual(counts,(21,3,1,2,1900))
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute('SELECT SUM(is_exact_duplicate) FROM transactions').fetchone()[0],1)
                self.assertEqual(conn.execute('SELECT SUM(class) FROM transactions').fetchone()[0],3)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transactions WHERE time_interval=0').fetchone()[0],21)

    def test_time_interval_split_at_14400(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'creditcard.csv'
            fixture(raw,[f'{i}.00' for i in range(20)], [0]*20,
                    [0,14400,28800,*([3600]*17)])
            db=root/'db.sqlite'
            run(raw,db,root/'out',SQL_DIR)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute('SELECT time_interval FROM transactions WHERE original_row=1').fetchone()[0],1)
                self.assertEqual(conn.execute('SELECT time_interval FROM transactions WHERE original_row=2').fetchone()[0],2)

    def test_bad_header_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'creditcard.csv'
            fixture(raw,[f'{i}.00' for i in range(20)],[0]*20)
            data=raw.read_text().replace('"Time"','"Clock"',1).replace('Time,','Clock,',1)
            raw.write_text(data)
            with self.assertRaises(ValueError):
                run(raw,root/'db.sqlite',root/'out',SQL_DIR)

    def test_build_connection_closed_before_replace(self):
        """Regression: Windows will not rename a .building DB with an open handle."""
        from src import sql_pipeline

        class TrackedConnection(sqlite3.Connection):
            is_closed = False

            def close(self):
                self.is_closed = True
                return super().close()

        original_connect = sqlite3.connect
        original_replace = sql_pipeline.os.replace
        building_connections = []

        def tracked_connect(database, *args, **kwargs):
            if str(database).endswith('.building'):
                kwargs['factory'] = TrackedConnection
                conn = original_connect(database, *args, **kwargs)
                building_connections.append(conn)
                return conn
            return original_connect(database, *args, **kwargs)

        def guarded_replace(source, target):
            if str(source).endswith('.building'):
                self.assertTrue(building_connections)
                self.assertTrue(all(conn.is_closed for conn in building_connections),
                                'SQLite connection must close before os.replace on Windows')
            return original_replace(source, target)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / 'creditcard.csv'
            fixture(raw, [f'{v}.00' for v in range(20)], [int(v == 19) for v in range(20)])
            with patch.object(sql_pipeline.sqlite3, 'connect', side_effect=tracked_connect), \
                 patch.object(sql_pipeline.os, 'replace', side_effect=guarded_replace):
                run(raw, root / 'db.sqlite', root / 'out', SQL_DIR)
            self.assertTrue(all(conn.is_closed for conn in building_connections))

    def test_source_unchanged_and_rerunnable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'creditcard.csv'; db=root/'db.sqlite'; out=root/'out'
            fixture(raw,[f'{v}.00' for v in range(20)],[int(v%4==0) for v in range(20)])
            before=raw.read_bytes()
            run(raw,db,out,SQL_DIR)
            run(raw,db,out,SQL_DIR)
            self.assertEqual(raw.read_bytes(),before)
            self.assertTrue((out/'sql_audit.json').exists())

if __name__=='__main__':
    unittest.main()
