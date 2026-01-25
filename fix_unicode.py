"""
Fix Unicode encoding issues in all Python test files
Replaces Unicode characters with ASCII-safe alternatives
"""

import os
import re

# Define replacements
replacements = {
    '[OK]': '[OK]',
    '[FAIL]': '[FAIL]',
    '[ERROR]': '[ERROR]',
    '[SUCCESS]': '[SUCCESS]',
    '[WARNING]': '[WARNING]',
    '[CRITICAL]': '[CRITICAL]',
    '[HIGH]': '[HIGH]',
    '[MODERATE]': '[MODERATE]',
    '=': '=',
    '=': '=',
    '|': '|',
    '=': '=',
    '=': '=',
    '>': '>',
    '||': '||',
}

def fix_file(filepath):
    """Fix Unicode characters in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace each Unicode character
        for unicode_char, ascii_char in replacements.items():
            content = content.replace(unicode_char, ascii_char)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in the project."""
    print("Fixing Unicode encoding issues in Python files...\n")

    fixed_count = 0
    checked_count = 0

    # Walk through all Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', 'venv', '.pytest_cache']):
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                checked_count += 1

                if fix_file(filepath):
                    print(f"[OK] Fixed: {filepath}")
                    fixed_count += 1

    print(f"\n" + "=" * 70)
    print(f"Checked {checked_count} Python files")
    print(f"Fixed {fixed_count} files with Unicode issues")
    print("=" * 70)

    if fixed_count > 0:
        print("\nAll Unicode characters have been replaced with ASCII-safe alternatives.")
        print("Tests should now run without encoding errors on Windows.")
    else:
        print("\nNo files needed fixing.")

if __name__ == "__main__":
    main()

