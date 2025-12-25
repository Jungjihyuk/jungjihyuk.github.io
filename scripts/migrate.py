#!/usr/bin/env python3
"""
Hugo Migration Script
Monitors pending/posts/ and converts Hexo frontmatter to Hugo format
"""

import os
import re
import yaml
import shutil
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

class MigrationConfig:
    """Configuration for migration paths"""
    BASE_DIR = Path("/Users/jihyeokjeong/Documents/jihyeok/blog")
    PENDING_DIR = BASE_DIR / "pending" / "posts"
    CONTENT_DIR = BASE_DIR / "hugo-blog" / "content" / "posts"
    LOG_DIR = BASE_DIR / "hugo-blog" / "migration-logs"

class FrontmatterConverter:
    """Converts Hexo frontmatter to Hugo format"""

    @staticmethod
    def extract_frontmatter(content: str) -> tuple:
        """Extract frontmatter and body from markdown content"""
        pattern = r'^---\n(.*?)\n---\n(.*)'
        match = re.match(pattern, content, re.DOTALL)

        if match:
            frontmatter_str = match.group(1)
            body = match.group(2)
            frontmatter = yaml.safe_load(frontmatter_str)
            return frontmatter, body

        return None, content

    @staticmethod
    def convert_to_hugo_frontmatter(hexo_fm: dict, filename: str) -> dict:
        """Convert Hexo frontmatter to Hugo format"""
        hugo_fm = {}

        # Required fields
        hugo_fm['title'] = hexo_fm.get('title', 'Untitled')
        hugo_fm['date'] = hexo_fm.get('date', datetime.now().strftime('%Y-%m-%d'))
        hugo_fm['draft'] = False

        # Description (from excerpt)
        if 'excerpt' in hexo_fm:
            hugo_fm['description'] = hexo_fm['excerpt']

        # Categories (preserve as list)
        if 'categories' in hexo_fm:
            categories = hexo_fm['categories']
            hugo_fm['categories'] = categories if isinstance(categories, list) else [categories]

        # Tags (preserve as list)
        if 'tags' in hexo_fm:
            tags = hexo_fm['tags']
            hugo_fm['tags'] = tags if isinstance(tags, list) else [tags]

        # Slug (from filename, without .md extension)
        slug = Path(filename).stem
        hugo_fm['slug'] = slug

        return hugo_fm

    @staticmethod
    def generate_hugo_content(hugo_fm: dict, body: str) -> str:
        """Generate complete Hugo markdown file"""
        # Convert frontmatter to YAML
        fm_yaml = yaml.dump(hugo_fm,
                           default_flow_style=False,
                           allow_unicode=True,
                           sort_keys=False)

        # Assemble complete content
        hugo_content = f"---\n{fm_yaml}---\n{body}"

        return hugo_content

class MigrationHandler(FileSystemEventHandler):
    """Handles file system events for migration"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.converter = FrontmatterConverter()

    def on_created(self, event):
        """Process newly added markdown files"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return

        # Wait a bit to ensure file is fully written
        time.sleep(0.5)

        self.process_file(Path(event.src_path))

    def process_file(self, file_path: Path):
        """Process a single markdown file"""
        try:
            print(f"\n[INFO] Processing: {file_path.name}")

            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract and convert frontmatter
            hexo_fm, body = self.converter.extract_frontmatter(content)

            if hexo_fm is None:
                print(f"[WARN] No frontmatter found in {file_path.name}")
                return

            hugo_fm = self.converter.convert_to_hugo_frontmatter(hexo_fm, file_path.name)
            hugo_content = self.converter.generate_hugo_content(hugo_fm, body)

            # Write to content/posts/
            output_path = self.config.CONTENT_DIR / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(hugo_content)

            # Log transformation
            self.log_migration(file_path.name, hexo_fm, hugo_fm)

            print(f"[SUCCESS] Migrated to: {output_path}")

            # Remove from pending
            file_path.unlink()
            print(f"[INFO] Removed from pending: {file_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path.name}: {str(e)}")

    def log_migration(self, filename: str, hexo_fm: dict, hugo_fm: dict):
        """Log migration details"""
        log_file = self.config.LOG_DIR / f"migration-{datetime.now().strftime('%Y%m%d')}.log"

        self.config.LOG_DIR.mkdir(exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"File: {filename}\n")
            f.write(f"Hexo Frontmatter:\n{yaml.dump(hexo_fm, allow_unicode=True)}\n")
            f.write(f"Hugo Frontmatter:\n{yaml.dump(hugo_fm, allow_unicode=True)}\n")

def main():
    """Main migration script"""
    config = MigrationConfig()

    # Ensure directories exist
    config.PENDING_DIR.mkdir(parents=True, exist_ok=True)
    config.CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Hugo Migration Script - Monitoring Mode")
    print("="*60)
    print(f"Watching: {config.PENDING_DIR}")
    print(f"Output:   {config.CONTENT_DIR}")
    print("\nCopy markdown files from Origin-blog to pending/posts/")
    print("They will be automatically converted and moved to content/posts/")
    print("\nPress Ctrl+C to stop\n")

    # Setup file system observer
    event_handler = MigrationHandler(config)
    observer = Observer()
    observer.schedule(event_handler, str(config.PENDING_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n\n[INFO] Migration script stopped")

    observer.join()

if __name__ == "__main__":
    main()
