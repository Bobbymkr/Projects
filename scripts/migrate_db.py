#!/usr/bin/env python3
"""
Database Migration Script.

Runs Alembic migrations for database schema management.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from alembic.config import Config
from alembic import command


def run_migrations(direction: str = "upgrade", revision: str = "head"):
    """
    Run database migrations.
    
    Args:
        direction: Migration direction ('upgrade' or 'downgrade')
        revision: Target revision (default: 'head' for upgrade, '-1' for downgrade)
    """
    alembic_cfg = Config(str(project_root / "src" / "api" / "database" / "migrations" / "alembic.ini"))
    
    # Override sqlalchemy.url from environment
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    if direction == "upgrade":
        print(f"Upgrading database to revision: {revision}")
        command.upgrade(alembic_cfg, revision)
    elif direction == "downgrade":
        print(f"Downgrading database to revision: {revision}")
        command.downgrade(alembic_cfg, revision)
    elif direction == "current":
        command.current(alembic_cfg)
    elif direction == "history":
        command.history(alembic_cfg)
    else:
        print(f"Unknown direction: {direction}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Database Migration Tool")
    parser.add_argument(
        "action",
        choices=["upgrade", "downgrade", "current", "history"],
        help="Migration action",
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target revision (default: 'head' for upgrade, '-1' for downgrade)",
    )
    
    args = parser.parse_args()
    
    if args.action == "downgrade" and args.revision == "head":
        args.revision = "-1"
    
    run_migrations(args.action, args.revision)


if __name__ == "__main__":
    main()

