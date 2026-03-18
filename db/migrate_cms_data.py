#!/usr/bin/env python3
"""
HerdSync CMS Data Migration

Imports old CMS CSV exports into the new Postgres schema.
Run after the herdsync-db container has started (it applies schema.sql on boot).

Usage:
  python3 migrate_cms_data.py \
    --host your-rds-host.amazonaws.com \
    --db herdsync \
    --user herdsync_admin \
    --password 'yourpassword' \
    --chickens cms_chicken_records.csv \
    --goats cms_goat_records.csv \
    --lambs cms_lamb_records.csv \
    --dry-run
"""

import csv
import argparse
import os
import sys
from datetime import datetime

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def parse_date(val):
    if not val or not val.strip():
        return None
    val = val.strip()
    for fmt in ('%m/%d/%Y', '%m/%d/%y'):
        try:
            return datetime.strptime(val, fmt).date()
        except ValueError:
            continue
    print(f"  WARNING: Could not parse date '{val}'")
    return None


def parse_weight(val):
    if not val or not val.strip():
        return None
    try:
        w = float(val.strip())
        return w if w > 0 else None
    except ValueError:
        return None


def extract_providers(csv_paths):
    """Extract unique providers from all CSVs, keyed by old farm_id."""
    providers = {}
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                farm_id = row.get('provider_farm_id', '').strip()
                if not farm_id:
                    continue
                if farm_id not in providers:
                    providers[farm_id] = {
                        'name': row.get('provider', '').strip(),
                        'phone': row.get('provider_phone', '').strip() or row.get('provider_mobile', '').strip(),
                    }
    return providers


def migrate(conn, chickens_path, goats_path, lambs_path, dry_run=False):
    cur = conn.cursor()

    # --- Providers ---
    print("\n=== PROVIDERS ===")
    all_paths = [p for p in [chickens_path, goats_path, lambs_path] if p]
    providers = extract_providers(all_paths)
    print(f"Found {len(providers)} unique providers")

    # Insert and build farm_id -> db_id mapping (in-memory only)
    farm_id_to_db_id = {}
    for farm_id, prov in providers.items():
        cur.execute(
            """INSERT INTO providers (name, phone, status)
               VALUES (%s, %s, 'active')
               RETURNING id""",
            (prov['name'], prov['phone']),
        )
        farm_id_to_db_id[farm_id] = cur.fetchone()[0]

    print(f"Inserted {len(farm_id_to_db_id)} providers")

    # --- Chickens ---
    if chickens_path and os.path.exists(chickens_path):
        print("\n=== CHICKENS ===")
        imported = 0
        skipped = 0
        with open(chickens_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                serial_id = row.get('serial_number', '').strip()
                if not serial_id or not serial_id.isdigit():
                    skipped += 1
                    continue

                prov_id = farm_id_to_db_id.get(row.get('provider_farm_id', '').strip())
                hang_weight = parse_weight(row.get('hang_weight', ''))

                cur.execute(
                    """INSERT INTO chickens (serial_id, hang_weight, whole_hw, prov_id, kill_date, process_date)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (serial_id) DO NOTHING""",
                    (
                        int(serial_id), hang_weight, hang_weight, prov_id,
                        parse_date(row.get('kill_date', '')),
                        parse_date(row.get('processing_date', '')),
                    ),
                )
                imported += 1
        print(f"Imported {imported}, skipped {skipped}")

    # --- Goats ---
    if goats_path and os.path.exists(goats_path):
        print("\n=== GOATS ===")
        imported = 0
        skipped = 0
        with open(goats_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                serial_id = row.get('serial_number', '').strip()
                if not serial_id or not serial_id.isdigit():
                    skipped += 1
                    continue

                prov_id = farm_id_to_db_id.get(row.get('provider_farm_id', '').strip())
                hang_weight = parse_weight(row.get('hang_weight_display', '') or row.get('proc_weight', ''))
                whole_hw = parse_weight(row.get('whole_hang_weight_display', '') or row.get('proc_wholeweight', ''))

                cur.execute(
                    """INSERT INTO goats (serial_id, hook_id, description, hang_weight, whole_hw, prov_id, kill_date, process_date)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (serial_id) DO NOTHING""",
                    (
                        int(serial_id),
                        row.get('hook_id', '').strip() or None,
                        row.get('description', '').strip() or None,
                        hang_weight, whole_hw, prov_id,
                        parse_date(row.get('kill_date', '')),
                        parse_date(row.get('processing_date', '')),
                    ),
                )
                imported += 1
        print(f"Imported {imported}, skipped {skipped}")

    # --- Lambs ---
    if lambs_path and os.path.exists(lambs_path):
        print("\n=== LAMBS ===")
        imported = 0
        skipped = 0
        with open(lambs_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                serial_id = row.get('serial_number', '').strip()
                if not serial_id or not serial_id.isdigit():
                    skipped += 1
                    continue

                prov_id = farm_id_to_db_id.get(row.get('provider_farm_id', '').strip())
                hang_weight = parse_weight(row.get('hang_weight_display', '') or row.get('proc_weight', ''))
                whole_hw = parse_weight(row.get('whole_hang_weight_display', '') or row.get('proc_wholeweight', ''))

                cur.execute(
                    """INSERT INTO lambs (serial_id, description, hang_weight, whole_hw, prov_id, kill_date, process_date)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (serial_id) DO NOTHING""",
                    (
                        int(serial_id),
                        row.get('description', '').strip() or None,
                        hang_weight, whole_hw, prov_id,
                        parse_date(row.get('kill_date', '')),
                        parse_date(row.get('processing_date', '')),
                    ),
                )
                imported += 1
        print(f"Imported {imported}, skipped {skipped}")

    # --- Commit or rollback ---
    if dry_run:
        conn.rollback()
        print("\n*** DRY RUN — rolled back ***")
    else:
        conn.commit()
        print("\n*** Committed ***")

    # --- Verify ---
    print("\n=== COUNTS ===")
    for table in ['providers', 'chickens', 'goats', 'lambs']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]}")

    cur.close()


def main():
    parser = argparse.ArgumentParser(description='Import CMS data into HerdSync')
    parser.add_argument('--host', default=os.environ.get('PGHOST', 'localhost'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PGPORT', 5432)))
    parser.add_argument('--db', default=os.environ.get('PGDATABASE', 'herdsync'))
    parser.add_argument('--user', default=os.environ.get('PGUSER', 'herdsync_admin'))
    parser.add_argument('--password', default=os.environ.get('PGPASSWORD', ''))
    parser.add_argument('--chickens', help='Path to cms_chicken_records.csv')
    parser.add_argument('--goats', help='Path to cms_goat_records.csv')
    parser.add_argument('--lambs', help='Path to cms_lamb_records.csv')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        conn = psycopg2.connect(db_url, sslmode='require')
    else:
        conn = psycopg2.connect(
            host=args.host, port=args.port, dbname=args.db,
            user=args.user, password=args.password, sslmode='require',
        )

    print("Connected")
    try:
        migrate(conn, args.chickens, args.goats, args.lambs, dry_run=args.dry_run)
    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
