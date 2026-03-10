#!/usr/bin/env python3
"""Create the MLRL_os database and tables.

Usage:
    python scripts/create_db.py
    python scripts/create_db.py --database-url postgresql://user:pass@host:5432/MLRL_os
    python scripts/create_db.py --drop  # Drop and recreate all tables

Requires asyncpg. Tables are created idempotently (IF NOT EXISTS).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}',
    data_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT,
    problem_type TEXT NOT NULL,
    dataset_id TEXT REFERENCES datasets(id),
    config JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    algorithm TEXT NOT NULL,
    task TEXT NOT NULL,
    feature_names JSONB DEFAULT '[]',
    metrics JSONB NOT NULL DEFAULT '{}',
    artifact_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rl_policies (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    algorithm TEXT NOT NULL,
    template TEXT NOT NULL,
    reward_function TEXT NOT NULL,
    training_curves JSONB,
    artifact_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS rl_policies CASCADE;
DROP TABLE IF EXISTS models CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
"""

# Indices for common query patterns
CREATE_INDICES_SQL = """
CREATE INDEX IF NOT EXISTS idx_datasets_source_type ON datasets(source_type);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_problem_type ON experiments(problem_type);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset_id ON experiments(dataset_id);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_models_experiment_id ON models(experiment_id);
CREATE INDEX IF NOT EXISTS idx_models_algorithm ON models(algorithm);
CREATE INDEX IF NOT EXISTS idx_rl_policies_experiment_id ON rl_policies(experiment_id);
CREATE INDEX IF NOT EXISTS idx_rl_policies_template ON rl_policies(template);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def create_database(database_url: str, *, drop: bool = False) -> None:
    """Create the MLRL_os tables (and optionally drop them first).

    Args:
        database_url: PostgreSQL connection string.
        drop: If True, drop all tables before recreating.
    """
    try:
        import asyncpg
    except ImportError:
        logger.error(
            "asyncpg is required but not installed. "
            "Install it with: pip install asyncpg"
        )
        sys.exit(1)

    conn = await asyncpg.connect(database_url)
    try:
        if drop:
            logger.warning("Dropping all MLRL_os tables...")
            await conn.execute(DROP_TABLES_SQL)
            logger.info("Tables dropped successfully.")

        logger.info("Creating MLRL_os tables...")
        await conn.execute(CREATE_TABLES_SQL)
        logger.info("Tables created successfully.")

        logger.info("Creating indices...")
        await conn.execute(CREATE_INDICES_SQL)
        logger.info("Indices created successfully.")

        # Verify tables exist
        tables = await conn.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('datasets', 'experiments', 'models', 'rl_policies')
            ORDER BY table_name
            """
        )
        table_names = [row["table_name"] for row in tables]
        logger.info("Verified tables: %s", ", ".join(table_names))

        if len(table_names) != 4:
            logger.error(
                "Expected 4 tables, found %d. Missing: %s",
                len(table_names),
                set(["datasets", "experiments", "models", "rl_policies"]) - set(table_names),
            )
            sys.exit(1)

    finally:
        await conn.close()

    logger.info("MLRL_os database setup complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create MLRL_os database tables."
    )
    parser.add_argument(
        "--database-url",
        default="postgresql://mlrlos:mlrlos_dev@localhost:5432/MLRL_os",
        help="PostgreSQL connection string (default: %(default)s)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate all tables (WARNING: destroys data)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    asyncio.run(create_database(args.database_url, drop=args.drop))


if __name__ == "__main__":
    main()
