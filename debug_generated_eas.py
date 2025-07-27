#!/usr/bin/env python3
import json
from pathlib import Path

def test_generated_eas_loading():
    """Test the generated EAs loading logic"""
    
    def _extract_strategy_type(template_name):
        """Extract strategy type from template name"""
        if 'trend' in template_name.lower():
            return 'Trend Following'
        elif 'mean' in template_name.lower() or 'reversion' in template_name.lower():
            return 'Mean Reversion'
        elif 'scalping' in template_name.lower():
            return 'Scalping'
        elif 'breakout' in template_name.lower():
            return 'Breakout'
        else:
            return 'Mixed Strategy'
    
    generated_eas_dir = Path("qnti_generated_eas")
    if not generated_eas_dir.exists():
        print("❌ Directory qnti_generated_eas does not exist")
        return
    
    generated_eas = []
    file_count = 0
    
    # Load all generated EA files
    for ea_file in generated_eas_dir.glob("*.json"):
        file_count += 1
        print(f"🔄 Processing file {file_count}: {ea_file.name}")
        
        try:
            with open(ea_file, 'r', encoding='utf-8') as f:
                ea_data = json.load(f)
            
            print(f"  ✅ File loaded successfully")
            print(f"  📊 Keys in file: {list(ea_data.keys())}")
            
            # Extract performance metrics from the correct structure
            performance_metrics = ea_data.get('performance_metrics', {})
            print(f"  📈 Performance metrics keys: {list(performance_metrics.keys())}")
            
            # Extract key information for display
            ea_info = {
                "id": ea_data.get('ea_id', ea_file.stem),
                "name": ea_data.get('template_name', 'Unknown Strategy'),
                "strategy_type": _extract_strategy_type(ea_data.get('template_name', '')),
                "symbol": "GOLD",  # Default to GOLD as that's what we see in backtests
                "timeframe": "M15",  # Default to M15 as that's what we see in backtests
                "win_rate": float(performance_metrics.get('win_rate', 0.0)) * 100,  # Convert to percentage
                "profit_factor": float(performance_metrics.get('profit_factor', 0.0)),
                "total_trades": int(performance_metrics.get('total_trades', 0)),
                "max_drawdown": float(performance_metrics.get('max_drawdown', 0.0)) * 100,  # Convert to percentage
                "sharpe_ratio": float(performance_metrics.get('sharpe_ratio', 0.0)),
                "total_profit": float(performance_metrics.get('total_return', 0.0)) * 1000,  # Convert to dollars
                "validation_status": ea_data.get('validation_status', 'Unknown'),
                "backtest_period": "Historical Data",
                "created_at": ea_data.get('created_at', 'Unknown'),
                "parameters": ea_data.get('optimized_parameters', {}),
                "performance_summary": performance_metrics,
                "generation_time": ea_data.get('generation_time', 0.0)
            }
            
            generated_eas.append(ea_info)
            print(f"  ✅ EA added to list: {ea_info['name']}")
            
        except Exception as e:
            print(f"  ❌ Error loading generated EA file {ea_file}: {e}")
            continue
        
        if file_count >= 3:  # Only process first 3 files for testing
            break
    
    print(f"\n📊 Summary:")
    print(f"  - Total files processed: {file_count}")
    print(f"  - Successfully loaded EAs: {len(generated_eas)}")
    
    if generated_eas:
        print(f"\n🎯 Sample EA details:")
        for i, ea in enumerate(generated_eas[:2]):
            print(f"  EA {i+1}:")
            print(f"    Name: {ea['name']}")
            print(f"    Strategy: {ea['strategy_type']}")
            print(f"    Win Rate: {ea['win_rate']:.1f}%")
            print(f"    Profit Factor: {ea['profit_factor']:.2f}")
            print(f"    Validation: {ea['validation_status']}")
    
    return generated_eas

if __name__ == "__main__":
    test_generated_eas_loading() 