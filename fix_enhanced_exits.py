# -*- coding: utf-8 -*-
"""
🎯 Fix Enhanced Exits Variable
Simple targeted fix for the undefined "enhanced_exits" variable
"""

import os
import shutil
import datetime

def fix_enhanced_exits_variable(file_path: str) -> bool:
    """🎯 Fix the enhanced_exits variable definition issue"""
    try:
        # Create backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.enhanced_exits_backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        print(f"📁 Backup: {backup_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find all lines that use enhanced_exits
        usage_lines = []
        definition_exists = False
        
        for i, line in enumerate(lines):
            if 'enhanced_exits' in line:
                if '=' in line and line.find('enhanced_exits') < line.find('='):
                    # This is a definition
                    definition_exists = True
                    print(f"✅ Found definition at line {i+1}: {line.strip()}")
                else:
                    # This is usage
                    usage_lines.append(i)
                    print(f"🔍 Found usage at line {i+1}: {line.strip()}")
        
        if not definition_exists and usage_lines:
            # Add definition before first usage
            first_usage_line = usage_lines[0]
            indent = len(lines[first_usage_line]) - len(lines[first_usage_line].lstrip())
            
            # Create a comprehensive enhanced_exits definition
            definition_lines = [
                ' ' * indent + '# 🎯 Enhanced exit strategy configuration\n',
                ' ' * indent + 'enhanced_exits = {\n',
                ' ' * (indent + 4) + '"targets": [6, 12, 20, 30],  # Profit target percentages\n',
                ' ' * (indent + 4) + '"percentages": [15, 20, 25, 40],  # Percentage to sell at each target\n',
                ' ' * (indent + 4) + '"vix_adjusted": False,  # Whether VIX adjustments are applied\n',
                ' ' * (indent + 4) + '"current_vix": 20  # Current VIX level\n',
                ' ' * indent + '}\n',
                '\n'
            ]
            
            # Insert the definition
            for j, def_line in enumerate(definition_lines):
                lines.insert(first_usage_line + j, def_line)
            
            print(f"✅ Added enhanced_exits definition before line {first_usage_line + 1}")
            
            # Write the file back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
        
        elif definition_exists:
            print("✅ enhanced_exits is already defined")
            return True
        
        else:
            print("ℹ️  No usage of enhanced_exits found")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing enhanced_exits: {e}")
        return False

def test_syntax(file_path: str) -> bool:
    """🧪 Test if file compiles after fix"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, file_path, 'exec')
        print("✅ File compiles successfully!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def main():
    """🚀 Main function"""
    print("🎯 Fix Enhanced Exits Variable")
    print("=" * 35)
    
    file_path = r"C:\bot_wyckoff2\fractional_position_system.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print("🔍 Analyzing enhanced_exits usage...")
    success = fix_enhanced_exits_variable(file_path)
    
    if success:
        print("\n🧪 Testing syntax...")
        if test_syntax(file_path):
            print("\n🎉 Enhanced exits variable fixed!")
            print("✅ Ready for Signal Quality Enhancement!")
            return True
        else:
            print("\n⚠️ Still has syntax issues to resolve")
            return False
    else:
        print("\n❌ Failed to fix enhanced_exits")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n📈 Next up: Signal Quality Enhancement")
        print("🎯 Multi-timeframe Wyckoff confirmation system")
    else:
        print("\n🔧 Need to resolve remaining issues first")