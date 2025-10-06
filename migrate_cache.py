#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Migration Script
Consolidates cache files from multiple directories into unified cache location.
Run this once after implementing the cache directory fixes.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import json

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def migrate_cache():
    """Migrate all cache files to unified cache directory"""

    print("=" * 70)
    print("CACHE MIGRATION SCRIPT")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Define paths
    project_root = Path(__file__).parent
    unified_cache = project_root / "cache"

    old_cache_locations = [
        project_root / "stock_analyzer" / "cache",
        project_root / "core" / "cache",
        project_root / "core" / "config" / "cache"
    ]

    # Create unified cache directory
    unified_cache.mkdir(exist_ok=True)
    print(f"✓ Unified cache directory: {unified_cache}")

    # Track statistics
    stats = {
        'total_files': 0,
        'migrated': 0,
        'skipped': 0,
        'merged': 0,
        'errors': 0
    }

    # Migrate files from each old location
    for old_cache in old_cache_locations:
        if not old_cache.exists():
            print(f"⊗ Skipping (not found): {old_cache}")
            continue

        print(f"\n📁 Processing: {old_cache}")

        # Get all JSON files
        json_files = list(old_cache.glob("*.json"))
        stats['total_files'] += len(json_files)

        for old_file in json_files:
            try:
                new_file = unified_cache / old_file.name

                # Check if file already exists in unified cache
                if new_file.exists():
                    # Compare file sizes and timestamps
                    old_size = old_file.stat().st_size
                    new_size = new_file.stat().st_size
                    old_time = old_file.stat().st_mtime
                    new_time = new_file.stat().st_mtime

                    if old_time > new_time:
                        # Old file is newer, replace
                        shutil.copy2(old_file, new_file)
                        print(f"  ↻ Updated (newer): {old_file.name}")
                        stats['merged'] += 1
                    elif old_size > new_size and new_size < 500:
                        # New file is suspiciously small (likely placeholder), replace
                        shutil.copy2(old_file, new_file)
                        print(f"  ↻ Replaced (larger): {old_file.name}")
                        stats['merged'] += 1
                    else:
                        # Keep existing file
                        print(f"  - Kept existing: {old_file.name}")
                        stats['skipped'] += 1
                else:
                    # Copy new file
                    shutil.copy2(old_file, new_file)
                    print(f"  ✓ Migrated: {old_file.name}")
                    stats['migrated'] += 1

            except Exception as e:
                print(f"  ✗ Error with {old_file.name}: {e}")
                stats['errors'] += 1

    # Create backup info file
    backup_info = {
        'migration_date': datetime.now().isoformat(),
        'unified_cache_location': str(unified_cache),
        'old_cache_locations': [str(p) for p in old_cache_locations],
        'statistics': stats
    }

    backup_info_file = unified_cache / "_migration_info.json"
    with open(backup_info_file, 'w') as f:
        json.dump(backup_info, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"Total files found:     {stats['total_files']}")
    print(f"Migrated:              {stats['migrated']}")
    print(f"Updated/Merged:        {stats['merged']}")
    print(f"Skipped (existing):    {stats['skipped']}")
    print(f"Errors:                {stats['errors']}")
    print(f"\nUnified cache: {unified_cache}")
    print(f"\nBackup info saved to: {backup_info_file}")

    # Provide cleanup instructions
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Test your application with the unified cache")
    print("2. If everything works, you can safely delete old cache directories:")
    for old_cache in old_cache_locations:
        if old_cache.exists() and old_cache != unified_cache:
            print(f"   - {old_cache}")
    print("\n3. Or keep them as backup for now\n")

if __name__ == "__main__":
    try:
        migrate_cache()
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
