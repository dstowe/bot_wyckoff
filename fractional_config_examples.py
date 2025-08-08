#!/usr/bin/env python3
"""
Fractional Position Building Configuration Examples
Practical configurations for different account sizes and strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class AccountSizeConfig:
    """Configuration for different account sizes"""
    min_account: float
    max_account: float
    entry_range: Tuple[float, float]  # (min_trade, max_trade)
    max_additions: int
    min_balance_preserve: float
    description: str


@dataclass
class WyckoffPhaseConfig:
    """Configuration for Wyckoff phase behavior"""
    initial_allocation: float
    allow_additions: bool
    max_total_allocation: float
    description: str
    risk_level: str


class FractionalTradingConfigurations:
    """Pre-built configurations for different trading scenarios"""
    
    def __init__(self):
        self.configurations = {
            'conservative': self.get_conservative_config(),
            'balanced': self.get_balanced_config(),
            'aggressive': self.get_aggressive_config(),
            'small_account': self.get_small_account_config(),
            'large_account': self.get_large_account_config()
        }
    
    def get_conservative_config(self) -> Dict:
        """Conservative configuration for risk-averse trading"""
        return {
            'name': 'Conservative Fractional Building',
            'description': 'Lower risk, smaller positions, higher thresholds',
            
            'account_sizing': [
                AccountSizeConfig(0, 500, (3, 8), 2, 75, "Micro accounts - very small positions"),
                AccountSizeConfig(500, 1000, (5, 12), 3, 100, "Small accounts - conservative sizing"),
                AccountSizeConfig(1000, 2500, (8, 20), 3, 150, "Medium accounts - steady growth"),
                AccountSizeConfig(2500, float('inf'), (15, 35), 4, 250, "Larger accounts - conservative allocation")
            ],
            
            'wyckoff_phases': {
                'ST': WyckoffPhaseConfig(0.20, False, 0.20, 'Small test - 20% only', 'HIGH'),
                'Creek': WyckoffPhaseConfig(0.0, False, 0.0, 'No positions in consolidation', 'NEUTRAL'),
                'SOS': WyckoffPhaseConfig(0.40, True, 0.60, 'Main position on confirmed breakout', 'MEDIUM'),
                'LPS': WyckoffPhaseConfig(0.20, True, 0.80, 'Conservative completion', 'LOW'),
                'BU': WyckoffPhaseConfig(0.15, True, 0.80, 'Small opportunistic add', 'MEDIUM')
            },
            
            'signal_thresholds': {
                'min_strength': 0.6,  # Higher threshold
                'min_volume_confirmation': True,
                'min_wyckoff_score': 0.5,
                'require_sector_strength': True
            },
            
            'scaling_out': {
                'profit_targets': [
                    {'gain_pct': 0.08, 'sell_pct': 0.20, 'description': '8% gain: Take 20% profit'},
                    {'gain_pct': 0.15, 'sell_pct': 0.25, 'description': '15% gain: Take 25% more'},
                    {'gain_pct': 0.25, 'sell_pct': 0.30, 'description': '25% gain: Take 30% more'},
                ],
                'max_hold_period_days': 90,
                'distribution_exit_pct': 1.0
            },
            
            'risk_management': {
                'max_positions_total': 8,
                'max_positions_building': 4,
                'max_allocation_per_position': 0.15,  # 15% max per position
                'max_daily_trades': 3,
                'require_3_day_spacing': True
            }
        }
    
    def get_balanced_config(self) -> Dict:
        """Balanced configuration - your original specification"""
        return {
            'name': 'Balanced Fractional Building',
            'description': 'Balanced approach matching your original requirements',
            
            'account_sizing': [
                AccountSizeConfig(0, 500, (5, 15), 3, 50, "Small accounts - your original spec"),
                AccountSizeConfig(500, 1000, (10, 25), 4, 100, "Medium accounts - your original spec"),
                AccountSizeConfig(1000, 2000, (15, 35), 5, 150, "Larger accounts - your original spec"),
                AccountSizeConfig(2000, float('inf'), (25, 50), 6, 200, "Large accounts - your original spec")
            ],
            
            'wyckoff_phases': {
                'ST': WyckoffPhaseConfig(0.25, False, 0.25, 'Testing phase - start small', 'HIGH'),
                'Creek': WyckoffPhaseConfig(0.0, False, 0.0, 'Hold only - no additions', 'NEUTRAL'),
                'SOS': WyckoffPhaseConfig(0.50, True, 0.75, 'Main position on breakout', 'MEDIUM'),
                'LPS': WyckoffPhaseConfig(0.25, True, 1.0, 'Complete position on support test', 'LOW'),
                'BU': WyckoffPhaseConfig(0.25, True, 1.0, 'Add on pullback bounce', 'MEDIUM')
            },
            
            'signal_thresholds': {
                'min_strength': 0.4,  # Your original threshold
                'min_volume_confirmation': True,
                'min_wyckoff_score': 0.4,
                'require_sector_strength': False
            },
            
            'scaling_out': {
                'profit_targets': [
                    {'gain_pct': 0.10, 'sell_pct': 0.25, 'description': '10% gain: Take 25% profit'},
                    {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take another 25%'},
                    {'gain_pct': 0.35, 'sell_pct': 0.25, 'description': '35% gain: Take another 25%'},
                ],
                'max_hold_period_days': 120,
                'distribution_exit_pct': 1.0  # Exit remaining 25% on distribution
            },
            
            'risk_management': {
                'max_positions_total': 10,
                'max_positions_building': 6,
                'max_allocation_per_position': 0.20,  # 20% max per position
                'max_daily_trades': 5,
                'require_3_day_spacing': True
            }
        }
    
    def get_aggressive_config(self) -> Dict:
        """Aggressive configuration for growth-focused trading"""
        return {
            'name': 'Aggressive Fractional Building',
            'description': 'Higher risk, larger positions, more frequent trading',
            
            'account_sizing': [
                AccountSizeConfig(0, 500, (8, 20), 4, 25, "Micro accounts - aggressive sizing"),
                AccountSizeConfig(500, 1000, (15, 35), 5, 50, "Small accounts - growth focused"),
                AccountSizeConfig(1000, 2500, (25, 60), 6, 100, "Medium accounts - aggressive allocation"),
                AccountSizeConfig(2500, float('inf'), (40, 100), 8, 200, "Large accounts - maximum growth")
            ],
            
            'wyckoff_phases': {
                'ST': WyckoffPhaseConfig(0.35, False, 0.35, 'Larger test position', 'HIGH'),
                'Creek': WyckoffPhaseConfig(0.15, True, 0.50, 'Small adds in consolidation', 'NEUTRAL'),
                'SOS': WyckoffPhaseConfig(0.50, True, 0.85, 'Heavy allocation on breakout', 'MEDIUM'),
                'LPS': WyckoffPhaseConfig(0.30, True, 1.15, 'Over-allocate on strong support', 'LOW'),
                'BU': WyckoffPhaseConfig(0.35, True, 1.15, 'Large bounce additions', 'MEDIUM')
            },
            
            'signal_thresholds': {
                'min_strength': 0.3,  # Lower threshold for more opportunities
                'min_volume_confirmation': False,  # More flexible
                'min_wyckoff_score': 0.3,
                'require_sector_strength': False
            },
            
            'scaling_out': {
                'profit_targets': [
                    {'gain_pct': 0.12, 'sell_pct': 0.20, 'description': '12% gain: Take 20% profit'},
                    {'gain_pct': 0.25, 'sell_pct': 0.25, 'description': '25% gain: Take 25% more'},
                    {'gain_pct': 0.50, 'sell_pct': 0.30, 'description': '50% gain: Take 30% more'},
                ],
                'max_hold_period_days': 180,  # Hold longer for bigger gains
                'distribution_exit_pct': 1.0
            },
            
            'risk_management': {
                'max_positions_total': 15,
                'max_positions_building': 10,
                'max_allocation_per_position': 0.25,  # 25% max per position
                'max_daily_trades': 8,
                'require_3_day_spacing': False  # More frequent additions
            }
        }
    
    def get_small_account_config(self) -> Dict:
        """Optimized for accounts under $1000"""
        return {
            'name': 'Small Account Optimization',
            'description': 'Optimized for accounts under $1000 with fractional shares',
            
            'account_sizing': [
                AccountSizeConfig(0, 200, (2, 6), 2, 25, "Micro accounts - minimum viable"),
                AccountSizeConfig(200, 500, (4, 10), 3, 40, "Small accounts - careful growth"),
                AccountSizeConfig(500, 1000, (6, 18), 4, 60, "Medium-small accounts - steady building"),
                AccountSizeConfig(1000, float('inf'), (10, 25), 5, 100, "Graduating to medium account")
            ],
            
            'wyckoff_phases': {
                'ST': WyckoffPhaseConfig(0.30, False, 0.30, 'Meaningful test for small accounts', 'HIGH'),
                'Creek': WyckoffPhaseConfig(0.0, False, 0.0, 'No capital waste in consolidation', 'NEUTRAL'),
                'SOS': WyckoffPhaseConfig(0.45, True, 0.75, 'Primary allocation on confirmation', 'MEDIUM'),
                'LPS': WyckoffPhaseConfig(0.25, True, 1.0, 'Complete the position', 'LOW'),
                'BU': WyckoffPhaseConfig(0.20, True, 1.0, 'Opportunistic addition', 'MEDIUM')
            },
            
            'signal_thresholds': {
                'min_strength': 0.5,  # Quality over quantity for small accounts
                'min_volume_confirmation': True,
                'min_wyckoff_score': 0.4,
                'require_sector_strength': False
            },
            
            'scaling_out': {
                'profit_targets': [
                    {'gain_pct': 0.15, 'sell_pct': 0.30, 'description': '15% gain: Take 30% profit'},
                    {'gain_pct': 0.30, 'sell_pct': 0.35, 'description': '30% gain: Take 35% more'},
                    {'gain_pct': 0.50, 'sell_pct': 0.35, 'description': '50% gain: Take 35% more'},
                ],
                'max_hold_period_days': 60,  # Faster turnover for small accounts
                'distribution_exit_pct': 1.0
            },
            
            'risk_management': {
                'max_positions_total': 6,  # Fewer positions for focus
                'max_positions_building': 3,
                'max_allocation_per_position': 0.30,  # Higher concentration acceptable
                'max_daily_trades': 2,
                'require_3_day_spacing': True
            }
        }
    
    def get_large_account_config(self) -> Dict:
        """Optimized for accounts over $5000"""
        return {
            'name': 'Large Account Diversification',
            'description': 'Optimized for accounts over $5000 with diversification focus',
            
            'account_sizing': [
                AccountSizeConfig(0, 2000, (20, 40), 4, 200, "Medium baseline"),
                AccountSizeConfig(2000, 5000, (30, 75), 5, 300, "Large account entry"),
                AccountSizeConfig(5000, 10000, (50, 125), 6, 500, "Large account optimization"),
                AccountSizeConfig(10000, float('inf'), (75, 200), 8, 1000, "Large account maximum")
            ],
            
            'wyckoff_phases': {
                'ST': WyckoffPhaseConfig(0.20, False, 0.20, 'Conservative test for diversification', 'HIGH'),
                'Creek': WyckoffPhaseConfig(0.0, False, 0.0, 'Patience in consolidation', 'NEUTRAL'),
                'SOS': WyckoffPhaseConfig(0.40, True, 0.60, 'Measured allocation on breakout', 'MEDIUM'),
                'LPS': WyckoffPhaseConfig(0.25, True, 0.85, 'Conservative completion', 'LOW'),
                'BU': WyckoffPhaseConfig(0.20, True, 0.85, 'Measured bounce addition', 'MEDIUM')
            },
            
            'signal_thresholds': {
                'min_strength': 0.5,  # Quality focus for diversified portfolio
                'min_volume_confirmation': True,
                'min_wyckoff_score': 0.5,
                'require_sector_strength': True  # Sector rotation awareness
            },
            
            'scaling_out': {
                'profit_targets': [
                    {'gain_pct': 0.10, 'sell_pct': 0.20, 'description': '10% gain: Take 20% profit'},
                    {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take 25% more'},
                    {'gain_pct': 0.35, 'sell_pct': 0.30, 'description': '35% gain: Take 30% more'},
                    {'gain_pct': 0.60, 'sell_pct': 0.25, 'description': '60% gain: Final scaling'},
                ],
                'max_hold_period_days': 150,
                'distribution_exit_pct': 1.0
            },
            
            'risk_management': {
                'max_positions_total': 20,  # More diversification
                'max_positions_building': 12,
                'max_allocation_per_position': 0.10,  # Lower concentration
                'max_daily_trades': 6,
                'require_3_day_spacing': True
            }
        }
    
    def apply_configuration(self, config_name: str, bot_instance):
        """Apply a configuration to a bot instance"""
        if config_name not in self.configurations:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        config = self.configurations[config_name]
        
        # Apply account sizing
        bot_instance.position_manager.account_sizing_config = [
            {
                'min_account': cfg.min_account,
                'max_account': cfg.max_account,
                'entry_range': cfg.entry_range,
                'max_additions': cfg.max_additions,
                'min_balance_preserve': cfg.min_balance_preserve
            }
            for cfg in config['account_sizing']
        ]
        
        # Apply Wyckoff phase configuration
        bot_instance.position_manager.phase_allocation_config = {
            phase: {
                'initial_allocation': cfg.initial_allocation,
                'allow_additions': cfg.allow_additions,
                'max_total_allocation': cfg.max_total_allocation,
                'description': cfg.description,
                'risk_level': cfg.risk_level
            }
            for phase, cfg in config['wyckoff_phases'].items()
        }
        
        # Apply signal thresholds
        signal_config = config['signal_thresholds']
        bot_instance.min_signal_strength = signal_config['min_strength']
        
        # Apply scaling out configuration
        scaling_config = config['scaling_out']
        bot_instance.position_manager.scaling_out_config = {
            'profit_targets': scaling_config['profit_targets'],
            'distribution_signals': {
                'phases': ['PS', 'SC'],
                'sell_pct': scaling_config['distribution_exit_pct'],
                'description': 'Distribution signal: Exit remaining position'
            }
        }
        
        # Apply risk management
        risk_config = config['risk_management']
        bot_instance.max_positions_total = risk_config['max_positions_total']
        bot_instance.max_daily_trades = risk_config['max_daily_trades']
        
        print(f"✅ Applied '{config['name']}' configuration")
        print(f"   Description: {config['description']}")
        
        return True
    
    def generate_custom_config(self, account_value: float, risk_tolerance: str) -> Dict:
        """Generate a custom configuration based on account value and risk tolerance"""
        
        # Determine base configuration
        if account_value < 500:
            base_config = self.get_small_account_config()
        elif account_value < 5000:
            base_config = self.get_balanced_config()
        else:
            base_config = self.get_large_account_config()
        
        # Adjust for risk tolerance
        if risk_tolerance.lower() == 'conservative':
            # Reduce position sizes by 20%
            for size_config in base_config['account_sizing']:
                min_trade, max_trade = size_config.entry_range
                size_config.entry_range = (min_trade * 0.8, max_trade * 0.8)
                size_config.max_additions = max(1, size_config.max_additions - 1)
            
            # Increase signal thresholds
            base_config['signal_thresholds']['min_strength'] += 0.1
            base_config['signal_thresholds']['min_wyckoff_score'] += 0.1
            
        elif risk_tolerance.lower() == 'aggressive':
            # Increase position sizes by 25%
            for size_config in base_config['account_sizing']:
                min_trade, max_trade = size_config.entry_range
                size_config.entry_range = (min_trade * 1.25, max_trade * 1.25)
                size_config.max_additions = size_config.max_additions + 1
            
            # Decrease signal thresholds
            base_config['signal_thresholds']['min_strength'] = max(0.2, 
                base_config['signal_thresholds']['min_strength'] - 0.1)
            base_config['signal_thresholds']['min_wyckoff_score'] = max(0.2,
                base_config['signal_thresholds']['min_wyckoff_score'] - 0.1)
        
        return base_config
    
    def save_configuration(self, config_name: str, filename: str):
        """Save a configuration to a JSON file"""
        if config_name not in self.configurations:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        config = self.configurations[config_name]
        
        # Convert to serializable format
        serializable_config = self._make_serializable(config)
        
        with open(filename, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        print(f"✅ Configuration '{config_name}' saved to {filename}")
    
    def _make_serializable(self, config: Dict) -> Dict:
        """Convert configuration to JSON-serializable format"""
        serializable = config.copy()
        
        # Convert AccountSizeConfig objects
        if 'account_sizing' in serializable:
            serializable['account_sizing'] = [
                {
                    'min_account': cfg.min_account,
                    'max_account': cfg.max_account,
                    'entry_range': cfg.entry_range,
                    'max_additions': cfg.max_additions,
                    'min_balance_preserve': cfg.min_balance_preserve,
                    'description': cfg.description
                }
                for cfg in serializable['account_sizing']
            ]
        
        # Convert WyckoffPhaseConfig objects
        if 'wyckoff_phases' in serializable:
            serializable['wyckoff_phases'] = {
                phase: {
                    'initial_allocation': cfg.initial_allocation,
                    'allow_additions': cfg.allow_additions,
                    'max_total_allocation': cfg.max_total_allocation,
                    'description': cfg.description,
                    'risk_level': cfg.risk_level
                }
                for phase, cfg in serializable['wyckoff_phases'].items()
            }
        
        return serializable


# Example usage and testing
def main():
    """Demonstrate configuration usage"""
    print("🔧 FRACTIONAL POSITION BUILDING CONFIGURATIONS")
    print("="*60)
    
    configs = FractionalTradingConfigurations()
    
    # Show all available configurations
    print("\nAvailable Configurations:")
    for name, config in configs.configurations.items():
        print(f"  {name}: {config['description']}")
    
    # Example: Generate custom configuration
    print(f"\n📊 Custom Configuration Example:")
    custom_config = configs.generate_custom_config(account_value=750, risk_tolerance='balanced')
    print(f"Generated configuration for $750 account with balanced risk")
    
    # Show account sizing for this configuration
    print(f"\nAccount Sizing Tiers:")
    for tier in custom_config['account_sizing']:
        min_trade, max_trade = tier.entry_range
        print(f"  ${tier.min_account:,}-${tier.max_account:,}: ${min_trade}-${max_trade} per trade, max {tier.max_additions} additions")
    
    # Show Wyckoff phase allocations
    print(f"\nWyckoff Phase Allocations:")
    for phase, config in custom_config['wyckoff_phases'].items():
        print(f"  {phase}: {config['initial_allocation']:.0%} initial, max {config['max_total_allocation']:.0%} total")
    
    # Save example configurations
    output_dir = "config_examples"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for config_name in ['balanced', 'small_account', 'conservative']:
        filename = f"{output_dir}/{config_name}_config.json"
        configs.save_configuration(config_name, filename)
    
    print(f"\n✅ Example configurations saved to {output_dir}/")
    
    # Show how to apply to bot
    print(f"\n🤖 To apply a configuration to your bot:")
    print(f"```python")
    print(f"configs = FractionalTradingConfigurations()")
    print(f"configs.apply_configuration('balanced', my_bot)")
    print(f"```")


if __name__ == "__main__":
    main()