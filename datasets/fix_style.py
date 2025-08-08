#!/usr/bin/env python3
"""
Quick style fixer for Python files
Fixes common PEP8 issues while preserving HTML content
"""

import re


def fix_python_style(file_path):
    """Fix common Python style issues"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix trailing whitespace on non-HTML lines
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Keep HTML lines as-is, but fix trailing whitespace on others
        if not ('<' in line and '>' in line):
            line = line.rstrip()
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix f-strings without placeholders
    content = re.sub(r'print\(f"([^{]*?)"\)', r'print("\1")', content)
    
    # Fix bare except clauses
    content = re.sub(r'except:', 'except Exception:', content)
    
    # Remove blank lines with only whitespace
    content = re.sub(r'\n[ \t]+\n', '\n\n', content)
    
    # Save fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed style issues in {file_path}")


if __name__ == "__main__":
    fix_python_style("app.py")
    fix_python_style("create_powerbi_dataset.py")
