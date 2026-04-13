# DB Service - `herdsync-db`

Data access layer for the HerdSync platform. Single gateway to the PostgreSQL database - all reads and writes to animal, provider, grading, and audit data go through this service.

Runs as a Docker container on EC2 (port 8002). Not exposed publicly - only accessible from `db-proxy` on the internal Docker network.

## What It Does

- CRUD for animals (goats, lambs, chickens) with unified serial IDs across species
- Gapless committed serial assignment through a transactional counter
- Persisted new-animal grading sessions with one active pending workflow at a time
- Provider (farmer) management
- Grade result storage with upsert (one grade per animal, re-grading replaces)
- Audit log storage (written by db-proxy after successful mutations)
- Schema auto-applied on startup via `schema.sql`
- Startup duplicate precheck for `grade_results.serial_id` before schema/index changes are applied

## Architecture Decisions

**Unified serial IDs** - `serial_counters` owns the committed animal counter shared across all species. Creating a goat, lamb, or chicken allocates the next serial inside the same DB transaction that inserts the row, so abandoned new-animal grading sessions do not burn IDs. This means serial #42 is globally unique - there's no goat #42 and chicken #42.

**Generic CRUD factory** - `animal_factory.py` generates full CRUD routers dynamically based on table configuration. Chickens, goats, and lambs share the same code with different field sets. Adding a new species is one dict entry.

**Persisted grading sessions** - new-animal grading stores draft review data in `grading_sessions` under a draft `analysis_key`, keeps only one pending unsaved session active at a time, and finalizes the real `serial_id` only when the operator accepts the grade.

**Grade upsert** - POST to `/grading` is a single SQL `INSERT ... ON CONFLICT (serial_id) DO UPDATE` backed by a unique index on `grade_results(serial_id)`. This enforces the one-grade-per-animal rule without requiring the client to know whether to POST or PUT.

**Fail-closed duplicate detection** - on startup, the service checks for historical duplicate `grade_results.serial_id` rows and aborts with a readable report instead of silently applying the unique index over bad data.

**asyncpg** - direct async PostgreSQL driver, no ORM. Connection pool (2-10 connections) with SSL required for RDS.

## Directory Structure

```
db/
├── Dockerfile              # python:3.12-slim, exposes port 8002
├── requirements.txt        # Pinned: FastAPI, asyncpg, pydantic, boto3
├── main.py                 # FastAPI app, connection pool, route mounting, health check
├── schema.sql              # Full PostgreSQL schema (CREATE IF NOT EXISTS, idempotent)
├── migrate_cms_data.py     # One-time legacy CSV data migration script
└── routes/
    ├── __init__.py         # Shared get_conn helper (pool from app state)
    ├── animal_factory.py   # Generic CRUD router factory for species tables
    ├── animals.py          # Committed serial preview and cross-species queries
    ├── chickens.py         # Generated router (1 line: build_animal_router("chickens"))
    ├── goats.py            # Generated router (1 line: build_animal_router("goats"))
    ├── lambs.py            # Generated router (1 line: build_animal_router("lambs"))
    ├── providers.py        # Provider CRUD
    ├── grading.py          # Grade result CRUD plus persisted new-animal grading sessions
    ├── serials.py          # Transactional serial counter + pending-session gate helpers
    └── audit.py            # Audit log storage and retrieval (filterable)
```

## Database Schema

Core tables created idempotently on startup:

- `animals` - unified serial_id namespace, species enum (chicken/goat/lamb)
- `serial_counters` - transactional committed counters (currently `animals`)
- `chickens`, `goats`, `lambs` - species-specific fields, FK to animals(serial_id)
- `providers` - farmer/supplier records
- `grade_results` - YOLO measurements, grade, S3 keys, timing, warnings, `analysis_key` (unique per `serial_id`)
- `grading_sessions` - pending new-animal grading drafts, review payloads, lifecycle status
- `audit_logs` - mutation history with JSONB detail, timestamps, user attribution

Triggers auto-update `updated_at` on chickens, goats, lambs, providers, and grading sessions.

## Environment Variables

| Variable              | Required | Description                       |
| --------------------- | -------- | --------------------------------- |
| `DATABASE_URL`        | Yes      | PostgreSQL connection string      |
| `S3_CAPTURES_BUCKET`  | No       | For S3 cleanup on animal deletion |
| `S3_PROCESSED_BUCKET` | No       | For S3 cleanup on animal deletion |
