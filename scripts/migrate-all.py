#!/usr/bin/env python3
"""
One-time migration script for all Hexo posts to Hugo
"""

import os
import re
import yaml
from pathlib import Path
from datetime import datetime

# Paths
ORIGIN_POSTS = Path("/Users/jihyeokjeong/Documents/jihyeok/blog/Origin-blog/source/_posts")
HUGO_POSTS = Path("/Users/jihyeokjeong/Documents/jihyeok/blog/hugo-blog/content/posts")
LOG_DIR = Path("/Users/jihyeokjeong/Documents/jihyeok/blog/hugo-blog/migration-logs")

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

def migrate_file(file_path: Path) -> bool:
    """Migrate a single file"""
    try:
        print(f"Processing: {file_path.name}")

        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract and convert frontmatter
        hexo_fm, body = extract_frontmatter(content)

        if hexo_fm is None:
            print(f"  WARN: No frontmatter found in {file_path.name}")
            return False

        hugo_fm = convert_to_hugo_frontmatter(hexo_fm, file_path.name)
        hugo_content = generate_hugo_content(hugo_fm, body)

        # Write to Hugo posts
        output_path = HUGO_POSTS / file_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(hugo_content)

        print(f"  ✓ Migrated to {output_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return False

def main():
    """Main migration function"""
    print("=" * 60)
    print("Hugo Migration - One-time Script")
    print("=" * 60)
    print(f"Source: {ORIGIN_POSTS}")
    print(f"Target: {HUGO_POSTS}")
    print()

    # Ensure target directory exists
    HUGO_POSTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Get all markdown files
    md_files = list(ORIGIN_POSTS.glob("*.md"))
    print(f"Found {len(md_files)} markdown files\n")

    # Migrate each file
    success_count = 0
    failed_count = 0

    for md_file in md_files:
        if migrate_file(md_file):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total files: {len(md_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print()

    # Create log
    log_file = LOG_DIR / f"migration-all-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Migration completed at {datetime.now().isoformat()}\n")
        f.write(f"Total: {len(md_files)}, Success: {success_count}, Failed: {failed_count}\n")

    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    main()
