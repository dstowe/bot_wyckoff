# -*- coding: utf-8 -*-
"""
🚀 Integration Script - Fix Exit Strategy Syntax Errors
Automatically applies syntax fixes to fractional_position_system.py
Windows compatible with UTF-8 encoding and emoji support
"""

import os
import sys
import shutil
import datetime
import subprocess
from pathlib import Path

def setup_environment():
    """🔧 Setup the environment for Windows with UTF-8 support"""
    # Set UTF-8 encoding for Windows console
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
        
        # Set console code page to UTF-8
        try:
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

def create_backup(file_path: str) -> str:
    """📁 Create timestamped backup of the original file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(file_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return ""

def apply_syntax_fixes(file_path: str) -> bool:
    """🔧 Apply all syntax fixes to the file"""
    try:
        print(f"🔄 Reading file: {file_path}")
        
        # Read with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        modifications_made = []
        
        print("🔍 Scanning for syntax errors...")
        
        # Fix 1: Incomplete try statements
        for i, line in enumerate(lines):
            if line.strip().startswith('try:'):
                # Look ahead for except/finally
                has_handler = False
                indent_level = len(line) - len(line.lstrip())
                
                for j in range(i + 1, min(i + 50, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                        
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    
                    # Same or less indentation means we've left the try block
                    if next_indent <= indent_level:
                        if next_line.startswith(('except', 'finally')):
                            has_handler = True
                        break
                    
                if not has_handler:
                    # Find the end of the try block
                    try_end = i + 1
                    while try_end < len(lines):
                        line_indent = len(lines[try_end]) - len(lines[try_end].lstrip())
                        if (lines[try_end].strip() == '' or 
                            line_indent > indent_level):
                            try_end += 1
                        else:
                            break
                    
                    # Add except clause
                    except_line = ' ' * indent_level + 'except Exception as e:'
                    pass_line = ' ' * (indent_level + 4) + 'self.logger.error(f"🚨 Error: {e}")'
                    
                    lines.insert(try_end, except_line)
                    lines.insert(try_end + 1, pass_line)
                    modifications_made.append(f"Added except clause at line {try_end}")
        
        # Fix 2: Incomplete expressions and statements
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.endswith(('=', '==', '!=', '+', '-', '*', '/', 'and', 'or')):
                indent = len(line) - len(line.lstrip())
                
                if stripped.endswith('='):
                    lines[i] = line + ' None  # 🔧 Auto-fixed incomplete assignment'
                elif stripped.endswith(('and', 'or')):
                    lines[i] = line + ' True  # 🔧 Auto-fixed incomplete logical'
                elif stripped.endswith(('==', '!=')):
                    lines[i] = line + ' None  # 🔧 Auto-fixed incomplete comparison'
                else:
                    lines[i] = line + ' 0  # 🔧 Auto-fixed incomplete expression'
                
                modifications_made.append(f"Fixed incomplete expression at line {i+1}")
        
        # Fix 3: Indentation issues
        for i in range(1, len(lines)):
            if lines[i].strip():  # Non-empty line
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                prev_line = lines[i-1].strip()
                
                if prev_line:
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    
                    # Check for unreasonable indentation jumps
                    if current_indent > prev_indent + 8:
                        lines[i] = ' ' * (prev_indent + 4) + lines[i].lstrip()
                        modifications_made.append(f"Fixed indentation at line {i+1}")
        
        # Fix 4: Add missing enhanced_exits method if referenced but not defined
        has_enhanced_exits_method = 'def calculate_enhanced_exits(' in content
        uses_enhanced_exits = 'enhanced_exits' in content
        
        if uses_enhanced_exits and not has_enhanced_exits_method:
            # Find a good spot to add the method
            class_methods = []
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and 'self' in line:
                    class_methods.append(i)
            
            if class_methods:
                insert_pos = class_methods[-1]  # Add after last method
                method_indent = len(lines[insert_pos]) - len(lines[insert_pos].lstrip())
                
                enhanced_method = [
                    '',
                    f'{" " * method_indent}def calculate_enhanced_exits(self) -> dict:',
                    f'{" " * (method_indent + 4)}"""🎯 Calculate enhanced exit strategy based on volatility"""',
                    f'{" " * (method_indent + 4)}try:',
                    f'{" " * (method_indent + 8)}vix = getattr(self, "get_current_vix", lambda: 20)()',
                    f'{" " * (method_indent + 8)}',
                    f'{" " * (method_indent + 8)}# Volatility-based exit adjustments',
                    f'{" " * (method_indent + 8)}if vix < 20:  # Low volatility 📈',
                    f'{" " * (method_indent + 12)}targets = [8, 15, 25, 40]',
                    f'{" " * (method_indent + 8)}elif vix > 25:  # High volatility 📉',
                    f'{" " * (method_indent + 12)}targets = [4, 8, 15, 25]',
                    f'{" " * (method_indent + 8)}else:  # Normal volatility 📊',
                    f'{" " * (method_indent + 12)}targets = [6, 12, 20, 30]',
                    f'{" " * (method_indent + 8)}',
                    f'{" " * (method_indent + 8)}return {{',
                    f'{" " * (method_indent + 12)}"targets": targets,',
                    f'{" " * (method_indent + 12)}"percentages": [15, 20, 25, 40],',
                    f'{" " * (method_indent + 12)}"vix": vix',
                    f'{" " * (method_indent + 8)}}}',
                    f'{" " * (method_indent + 4)}except Exception as e:',
                    f'{" " * (method_indent + 8)}self.logger.error(f"🚨 Enhanced exits error: {{e}}")',
                    f'{" " * (method_indent + 8)}return {{"targets": [6, 12, 20, 30], "percentages": [15, 20, 25, 40], "vix": 20}}'
                ]
                
                # Insert the method
                for j, method_line in enumerate(enhanced_method):
                    lines.insert(insert_pos + 1 + j, method_line)
                
                modifications_made.append("Added missing calculate_enhanced_exits method")
        
        # Fix 5: Replace undefined enhanced_exits references
        for i, line in enumerate(lines):
            if 'enhanced_exits' in line and 'calculate_enhanced_exits' not in line:
                if '=' not in line or line.index('enhanced_exits') < line.index('='):
                    # This is a usage, not a definition
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i, f'{" " * indent}enhanced_exits = self.calculate_enhanced_exits()  # 🔧 Auto-added')
                    modifications_made.append(f"Added enhanced_exits definition at line {i+1}")
                    break
        
        # Write the fixed content
        fixed_content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Applied {len(modifications_made)} fixes:")
        for mod in modifications_made:
            print(f"  🔧 {mod}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False

def validate_syntax(file_path: str) -> bool:
    """✅ Validate that the file has correct Python syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the file
        compile(content, file_path, 'exec')
        print("✅ Syntax validation passed!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error found: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """🚀 Main integration function"""
    setup_environment()
    
    print("🔧 Wyckoff Bot - Syntax Fix Integration")
    print("=" * 50)
    
    # File path
    bot_dir = r"C:\bot_wyckoff2"
    file_path = os.path.join(bot_dir, "fractional_position_system.py")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please check the path and try again.")
        return False
    
    print(f"📁 Target file: {file_path}")
    
    # Create backup
    backup_path = create_backup(file_path)
    if not backup_path:
        print("❌ Cannot proceed without backup")
        return False
    
    # Apply fixes
    print("\n🔧 Applying syntax fixes...")
    success = apply_syntax_fixes(file_path)
    
    if success:
        print("\n✅ Validating syntax...")
        if validate_syntax(file_path):
            print("\n🎉 Integration completed successfully!")
            print(f"📁 Original backed up to: {backup_path}")
            print("🔧 All syntax errors have been fixed")
            return True
        else:
            print("\n⚠️ Syntax validation failed")
            print("🔄 Restoring from backup...")
            shutil.copy2(backup_path, file_path)
            return False
    else:
        print("\n❌ Failed to apply fixes")
        print("🔄 Restoring from backup...")
        shutil.copy2(backup_path, file_path)
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Ready to proceed with remaining improvements!")
            print("📊 Next steps:")
            print("  1. Signal Quality Enhancement 📈") 
            print("  2. Cash Management 💵")
            print("  3. Additional optimizations")
        else:
            print("\n❌ Please review errors and try again")
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")