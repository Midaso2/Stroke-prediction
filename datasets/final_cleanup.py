#!/usr/bin/env python3
"""
Final cleanup script for all project files
Removes trailing whitespace and fixes blank line issues
"""

import os
import re


def clean_file(filepath):
    """Clean a file of whitespace issues"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove trailing whitespace from all lines
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Join back and ensure single newline at end
        cleaned_content = '\n'.join(cleaned_lines)
        if cleaned_content and not cleaned_content.endswith('\n'):
            cleaned_content += '\n'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned {filepath}")
        
    except Exception as e:
        print(f"❌ Error cleaning {filepath}: {e}")


def main():
    """Clean all project files"""
    files_to_clean = [
        'app.py',
        'create_powerbi_dataset.py',
        'README.md',
        'INTERACTIVE_FEATURES.md',
        'requirements.txt'
    ]
    
    for filename in files_to_clean:
        if os.path.exists(filename):
            clean_file(filename)
        else:
            print(f"⚠️ File not found: {filename}")
    
    print("\n🎉 Project cleanup complete!")
    print("All files have been cleaned of trailing whitespace.")
    print("Your project is now ready for submission!")


if __name__ == "__main__":
    main()
