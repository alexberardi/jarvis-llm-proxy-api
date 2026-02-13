#!/usr/bin/env python3
"""Seed settings from environment variables into the database.

This script is intended for initial setup or migration. It reads all
defined settings from environment variables and populates the database.
Non-sensitive runtime config should live in settings; env is for bootstrap.

Usage:
    python scripts/seed_settings.py [--dry-run] [--force]

Options:
    --dry-run   Show what would be done without making changes
    --force     Overwrite existing database values with env values
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed settings from environment variables to database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing database values with env values",
    )
    args = parser.parse_args()

    from services.settings_service import (
        get_settings_service,
        SETTINGS_DEFINITIONS,
    )
    from jarvis_settings_client import coerce_value

    settings = get_settings_service()

    print(f"Found {len(SETTINGS_DEFINITIONS)} setting definitions")
    print()

    synced = 0
    skipped = 0
    overwritten = 0

    for definition in SETTINGS_DEFINITIONS:
        key = definition.key
        env_var = definition.env_fallback

        # Check if env var is set
        if not env_var:
            print(f"⚪ {key}: No env fallback defined, skipping")
            skipped += 1
            continue

        env_value = os.getenv(env_var)
        if env_value is None:
            print(f"⚪ {key}: Env var {env_var} not set, skipping")
            skipped += 1
            continue

        # Coerce value
        coerced_value = coerce_value(env_value, definition.value_type, definition.default)

        # Check if already in database
        current_value = settings.get(key)
        with settings._cache_lock:
            from_db = key in settings._cache and settings._cache[key].from_db

        if from_db and not args.force:
            print(f"🔵 {key}: Already in DB (use --force to overwrite)")
            skipped += 1
            continue

        if from_db and args.force:
            action = "overwrite"
            overwritten += 1
        else:
            action = "create"
            synced += 1

        if args.dry_run:
            print(f"🟡 {key}: Would {action} with value from {env_var}")
        else:
            success = settings.set(key, coerced_value)
            if success:
                print(f"🟢 {key}: {action.capitalize()}d from {env_var}")
            else:
                print(f"🔴 {key}: Failed to {action}")

    print()
    print(f"Summary: {synced} synced, {overwritten} overwritten, {skipped} skipped")
    if args.dry_run:
        print("(Dry run - no changes made)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
