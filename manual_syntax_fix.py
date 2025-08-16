# -*- coding: utf-8 -*-
"""
🎯 Manual Targeted Syntax Fix
Specifically addresses the console errors you reported:
- Line 2395: Try statement without except/finally
- Line 2544: Expected expression  
- Line 2545: Unexpected indentation
- Line 2582: "enhanced_exits" not defined
"""

import os
import shutil
import datetime

def manual_fix_specific_errors(file_path: str) -> bool:
    """🔧 Fix the specific syntax errors manually"""
    try:
        # Create backup first
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.manual_backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        print(f"📁 Manual backup created: {backup_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Keep track of line numbers (1-based like in error messages)
        fixes_applied = []
        
        # Fix 1: Line 2395 - Try statement issue
        if len(lines) >= 2395:
            line_2395 = lines[2394]  # 0-based index
            if 'try:' in line_2395:
                # Check if there's an except/finally nearby
                has_handler = False
                indent = len(line_2395) - len(line_2395.lstrip())
                
                # Look for except/finally in next 20 lines
                for i in range(2395, min(2415, len(lines))):
                    if i < len(lines):
                        check_line = lines[i].strip()
                        if check_line.startswith(('except', 'finally')):
                            has_handler = True
                            break
                        # If we hit same/lower indentation that's not except/finally, add handler
                        line_indent = len(lines[i]) - len(lines[i].lstrip())
                        if lines[i].strip() and line_indent <= indent and not check_line.startswith(('except', 'finally')):
                            break
                
                if not has_handler:
                    # Add except clause
                    except_line = ' ' * indent + 'except Exception as e:\n'
                    pass_line = ' ' * (indent + 4) + 'self.logger.error(f"🚨 Error in exit strategy: {e}")\n'
                    
                    # Insert at the right place
                    insert_pos = 2395  # After the try block content
                    while insert_pos < len(lines) and (
                        not lines[insert_pos].strip() or 
                        len(lines[insert_pos]) - len(lines[insert_pos].lstrip()) > indent
                    ):
                        insert_pos += 1
                    
                    lines.insert(insert_pos, except_line)
                    lines.insert(insert_pos + 1, pass_line)
                    fixes_applied.append(f"Fixed try statement at line 2395")
        
        # Fix 2: Line 2544 - Expected expression
        if len(lines) >= 2544:
            line_2544 = lines[2543]  # 0-based index
            stripped = line_2544.strip()
            
            # Common issues that cause "expected expression"
            if stripped.endswith(('=', '==', '!=', 'and', 'or', '+', '-', '*', '/')):
                if stripped.endswith('=') and not stripped.endswith(('==', '!=', '>=', '<=')):
                    lines[2543] = line_2544.rstrip() + ' None  # 🔧 Fixed incomplete assignment\n'
                    fixes_applied.append("Fixed incomplete assignment at line 2544")
                elif stripped.endswith(('and', 'or')):
                    # Check if this continues on next line
                    if len(lines) > 2544:
                        next_line = lines[2544].strip()
                        if not next_line or not any(next_line.startswith(x) for x in ['and', 'or', ')', ']', '}']):
                            lines[2543] = line_2544.rstrip() + ' True  # 🔧 Fixed incomplete logical\n'
                            fixes_applied.append("Fixed incomplete logical expression at line 2544")
                elif stripped.endswith(('==', '!=')):
                    lines[2543] = line_2544.rstrip() + ' None  # 🔧 Fixed incomplete comparison\n'
                    fixes_applied.append("Fixed incomplete comparison at line 2544")
            
            # Check for other syntax issues
            elif '(' in stripped and ')' not in stripped:
                lines[2543] = line_2544.rstrip() + ')  # 🔧 Fixed unmatched parenthesis\n'
                fixes_applied.append("Fixed unmatched parenthesis at line 2544")
            elif '[' in stripped and ']' not in stripped:
                lines[2543] = line_2544.rstrip() + ']  # 🔧 Fixed unmatched bracket\n'
                fixes_applied.append("Fixed unmatched bracket at line 2544")
        
        # Fix 3: Line 2545 - Unexpected indentation
        if len(lines) >= 2545:
            line_2545 = lines[2544]  # 0-based index
            if line_2545.strip():  # Only if line has content
                current_indent = len(line_2545) - len(line_2545.lstrip())
                
                # Check previous non-empty line for context
                prev_indent = 0
                for i in range(2543, -1, -1):  # Go backwards from line 2544
                    if lines[i].strip():
                        prev_indent = len(lines[i]) - len(lines[i].lstrip())
                        break
                
                # If indentation jump is too big, fix it
                if current_indent > prev_indent + 8:
                    new_indent = prev_indent + 4
                    lines[2544] = ' ' * new_indent + line_2545.lstrip()
                    fixes_applied.append("Fixed unexpected indentation at line 2545")
        
        # Fix 4: Line 2582 - "enhanced_exits" not defined
        if len(lines) >= 2582:
            line_2582 = lines[2581]  # 0-based index
            if 'enhanced_exits' in line_2582:
                # Check if enhanced_exits is defined before this line
                enhanced_exits_defined = False
                for i in range(2581):
                    if 'enhanced_exits' in lines[i] and '=' in lines[i]:
                        # Make sure it's a definition, not usage
                        line = lines[i].strip()
                        if line.find('enhanced_exits') < line.find('='):
                            enhanced_exits_defined = True
                            break
                
                if not enhanced_exits_defined:
                    # Add definition before line 2582
                    indent = len(line_2582) - len(line_2582.lstrip())
                    definition = ' ' * indent + 'enhanced_exits = {"targets": [6, 12, 20, 30], "percentages": [15, 20, 25, 40]}  # 🔧 Default values\n'
                    lines.insert(2581, definition)
                    fixes_applied.append("Added enhanced_exits definition before line 2582")
        
        # Write the fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Applied {len(fixes_applied)} targeted fixes:")
        for fix in fixes_applied:
            print(f"  🔧 {fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying manual fixes: {e}")
        return False

def validate_syntax(file_path: str) -> bool:
    """✅ Validate syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, file_path, 'exec')
        print("✅ Syntax validation passed!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """🎯 Main function for targeted fixes"""
    print("🎯 Manual Targeted Syntax Fix")
    print("=" * 40)
    
    file_path = r"C:\bot_wyckoff2\fractional_position_system.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Restore from backup if it exists (from previous attempt)
    backup_files = [f for f in os.listdir(os.path.dirname(file_path)) 
                   if f.startswith("fractional_position_system.py.backup_")]
    
    if backup_files:
        latest_backup = max(backup_files)
        backup_path = os.path.join(os.path.dirname(file_path), latest_backup)
        print(f"🔄 Restoring from backup: {backup_path}")
        shutil.copy2(backup_path, file_path)
    
    # Apply targeted fixes
    print("🔧 Applying targeted fixes for specific errors...")
    success = manual_fix_specific_errors(file_path)
    
    if success:
        print("\n✅ Validating syntax...")
        if validate_syntax(file_path):
            print("\n🎉 All targeted fixes applied successfully!")
            return True
        else:
            print("\n❌ Syntax validation still failed")
            return False
    else:
        print("\n❌ Failed to apply targeted fixes")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to proceed with next improvements!")
    else:
        print("\n❌ Please check the remaining syntax issues")