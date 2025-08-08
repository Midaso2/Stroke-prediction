#!/usr/bin/env python3
"""
Quick style fixer for Markdown files
Fixes common MD style issues
"""

import re


def fix_markdown_style(file_path):
    """Fix common Markdown style issues"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add blank lines around headings
    content = re.sub(r'\n(#{1,6}[^\n]*)\n', r'\n\n\1\n\n', content)
    
    # Add blank lines around lists
    content = re.sub(r'\n(-[^\n]*(?:\n-[^\n]*)*)\n', r'\n\n\1\n\n', content)
    
    # Remove trailing punctuation from headings
    content = re.sub(r'(#{1,6}[^:\n]*):(\s*\n)', r'\1\2', content)
    
    # Fix excessive blank lines (more than 2)
    content = re.sub(r'\n\n\n+', r'\n\n', content)
    
    # Save fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    
    print(f"✅ Fixed Markdown style issues in {file_path}")


if __name__ == "__main__":
    fix_markdown_style("INTERACTIVE_FEATURES.md")
