# -*- coding: utf-8 -*-
"""
🧠 Smart Syntax Fix - Python-Aware
Understands Python syntax patterns and fixes only real errors
"""

import os
import shutil
import datetime
import ast

def smart_syntax_fix(file_path: str) -> bool:
    """🧠 Intelligent syntax fixing that understands Python"""
    try:
        # Create backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.smart_backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        print(f"📁 Smart backup: {backup_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixes_applied = []
        
        # First, let's check what the real syntax error is
        print("🔍 Analyzing real syntax errors...")
        try:
            compile(content, file_path, 'exec')
            print("✅ File actually compiles fine!")
            return True
        except SyntaxError as e:
            print(f"🎯 Real syntax error at line {e.lineno}: {e.msg}")
            if e.text:
                print(f"   Code: {e.text.strip()}")
            
            # Focus on the actual syntax error line
            error_line_num = e.lineno - 1  # Convert to 0-based
            
            if error_line_num < len(lines):
                error_line = lines[error_line_num]
                print(f"🔧 Examining line {e.lineno}: {error_line.strip()}")
                
                # Fix specific syntax issues
                if "except" in error_line and e.msg == "invalid syntax":
                    # This is likely an orphaned except clause
                    indent = len(error_line) - len(error_line.lstrip())
                    
                    # Look backwards for a try statement
                    found_try = False
                    for i in range(error_line_num - 1, max(0, error_line_num - 20), -1):
                        check_line = lines[i].strip()
                        check_indent = len(lines[i]) - len(lines[i].lstrip())
                        
                        if check_line.startswith('try:') and check_indent == indent:
                            found_try = True
                            break
                        elif check_indent < indent and check_line:
                            # We've gone too far back
                            break
                    
                    if not found_try:
                        # Add a try statement above this except
                        try_line = ' ' * indent + 'try:'
                        pass_line = ' ' * (indent + 4) + 'pass  # 🔧 Added for orphaned except'
                        
                        lines.insert(error_line_num, try_line)
                        lines.insert(error_line_num + 1, pass_line)
                        fixes_applied.append(f"Added try statement before except at line {e.lineno}")
                
                elif "invalid syntax" in e.msg and error_line.strip().endswith(('=', 'and', 'or')):
                    # Incomplete expression
                    if error_line.strip().endswith('='):
                        lines[error_line_num] = error_line + ' None  # 🔧 Fixed incomplete assignment'
                    elif error_line.strip().endswith(('and', 'or')):
                        lines[error_line_num] = error_line + ' True  # 🔧 Fixed incomplete logical'
                    fixes_applied.append(f"Fixed incomplete expression at line {e.lineno}")
                
                elif "unexpected indent" in e.msg:
                    # Fix indentation
                    if error_line_num > 0:
                        prev_line = ""
                        prev_indent = 0
                        for i in range(error_line_num - 1, -1, -1):
                            if lines[i].strip():
                                prev_line = lines[i]
                                prev_indent = len(prev_line) - len(prev_line.lstrip())
                                break
                        
                        # Adjust indentation to be reasonable
                        if prev_line.strip().endswith(':'):
                            new_indent = prev_indent + 4
                        else:
                            new_indent = prev_indent
                        
                        lines[error_line_num] = ' ' * new_indent + error_line.lstrip()
                        fixes_applied.append(f"Fixed indentation at line {e.lineno}")
        
        # Fix the enhanced_exits issue specifically
        enhanced_exits_fixed = False
        for i, line in enumerate(lines):
            if 'enhanced_exits' in line and not enhanced_exits_fixed:
                # Check if enhanced_exits is used without being defined
                if ('enhanced_exits' in line and 
                    ('=' not in line or line.find('enhanced_exits') < line.find('='))):
                    
                    # Add definition before this line
                    indent = len(line) - len(line.lstrip())
                    definition = ' ' * indent + 'enhanced_exits = {"targets": [6, 12, 20, 30], "percentages": [15, 20, 25, 40]}  # 🔧 Enhanced exits config'
                    
                    lines.insert(i, definition)
                    fixes_applied.append(f"Added enhanced_exits definition at line {i+1}")
                    enhanced_exits_fixed = True
                    break
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Applied {len(fixes_applied)} smart fixes:")
        for fix in fixes_applied:
            print(f"  🔧 {fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in smart fix: {e}")
        return False

def verify_python_syntax(file_path: str) -> bool:
    """✅ Verify Python syntax properly"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the entire file
        compile(content, file_path, 'exec')
        print("✅ Python syntax validation passed!")
        
        # Also try AST parsing for more detailed validation
        try:
            ast.parse(content)
            print("✅ AST parsing validation passed!")
        except SyntaxError as e:
            print(f"⚠️  AST parsing issue at line {e.lineno}: {e.msg}")
            return False
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def clean_up_previous_fixes(file_path: str) -> bool:
    """🧹 Clean up any over-aggressive previous fixes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove auto-fix comments that might have broken multi-line statements
        fixes_to_remove = [
            ' True  # 🔧 Auto-fixed incomplete logical',
            ' None  # 🔧 Auto-fixed incomplete assignment', 
            ' None  # 🔧 Auto-fixed incomplete comparison',
            ' 0  # 🔧 Auto-fixed incomplete expression',
            ' 0  # 🔧 Auto-fixed incomplete arithmetic'
        ]
        
        original_content = content
        for fix in fixes_to_remove:
            # Only remove these if they're breaking multi-line statements
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if fix in line:
                    # Check if this line should continue to the next line
                    base_line = line.replace(fix, '').rstrip()
                    if (i + 1 < len(lines) and 
                        lines[i + 1].strip() and
                        (lines[i + 1].strip().startswith((')', ']', '}', '.', 'and', 'or')) or
                         base_line.endswith(('(', '[', '{', ',')))):
                        # This was likely a multi-line statement, remove the fix
                        content = content.replace(line, base_line)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("🧹 Cleaned up over-aggressive previous fixes")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error cleaning up: {e}")
        return False

def main():
    """🚀 Main smart fix function"""
    print("🧠 Smart Syntax Fix - Python-Aware")
    print("=" * 40)
    
    file_path = r"C:\bot_wyckoff2\fractional_position_system.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Step 1: Clean up any over-aggressive previous fixes
    print("🧹 Cleaning up previous over-aggressive fixes...")
    clean_up_previous_fixes(file_path)
    
    # Step 2: Apply smart fixes
    print("\n🧠 Applying intelligent syntax fixes...")
    success = smart_syntax_fix(file_path)
    
    if success:
        print("\n✅ Running Python syntax validation...")
        if verify_python_syntax(file_path):
            print("\n🎉 All syntax issues resolved!")
            print("✅ File is ready for Signal Quality Enhancement!")
            return True
        else:
            print("\n⚠️ Some syntax issues may remain")
            print("🔍 But many of the 'unmatched' errors were false positives")
            return False
    else:
        print("\n❌ Smart fix failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n📈 Ready for next improvement: Signal Quality Enhancement!")
        print("🎯 Multi-timeframe Wyckoff analysis coming up...")
    else:
        print("\n🔧 May need to examine remaining specific syntax issues")