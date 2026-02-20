#!/usr/bin/env python3
"""
Test imports from joint_quant module.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports from joint_quant...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Python path: {sys.path[:3]}")

# Test 1: Import the module
try:
    import joint_quant
    print("\n✓ Successfully imported joint_quant module")
    print(f"  Location: {joint_quant.__file__}")
except Exception as e:
    print(f"\n✗ Failed to import joint_quant: {e}")
    sys.exit(1)

# Test 2: Check __all__
try:
    print(f"\n✓ joint_quant.__all__ has {len(joint_quant.__all__)} items")
    print(f"  First 5 items: {joint_quant.__all__[:5]}")
except Exception as e:
    print(f"\n✗ Failed to access __all__: {e}")

# Test 3: Import config module
try:
    from joint_quant import config
    print("\n✓ Successfully imported joint_quant.config")
except Exception as e:
    print(f"\n✗ Failed to import config: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import JointQuantConfig
try:
    from joint_quant.config import JointQuantConfig
    print("\n✓ Successfully imported JointQuantConfig from joint_quant.config")
    print(f"  Type: {type(JointQuantConfig)}")
except Exception as e:
    print(f"\n✗ Failed to import JointQuantConfig from joint_quant.config: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Import from __init__
try:
    from joint_quant import JointQuantConfig
    print("\n✓ Successfully imported JointQuantConfig from joint_quant")
    print(f"  Type: {type(JointQuantConfig)}")
    
    # Try to create an instance
    config = JointQuantConfig()
    print(f"  Successfully created instance: {config.model_name}")
except Exception as e:
    print(f"\n✗ Failed to import JointQuantConfig from joint_quant: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Import all required items
required_imports = [
    'JointQuantConfig', 'DEV', 'set_seed', 'cleanup_memory',
    'get_model', 'get_tokenizer', 'RMSN',
    'get_orthogonal_matrix', 'QuantizedLinear',
    'fuse_layer_norms', 'rotate_model',
    'compute_smooth_scale',
    'get_wikitext2', 'get_calibration_data',
    'evaluate_perplexity', 'evaluate_perplexity_simple',
    'train_r1_joint', 'train_r2_independent',
    'R1Module', 'R2Module', 'JointR1Module',
    'collect_activations', 'whip_loss',
]

failed_imports = []
for name in required_imports:
    try:
        item = getattr(joint_quant, name)
        # print(f"  ✓ {name}")
    except AttributeError:
        failed_imports.append(name)
        print(f"  ✗ {name}")

if failed_imports:
    print(f"\n✗ Failed to import {len(failed_imports)} items: {failed_imports}")
else:
    print(f"\n✓ All {len(required_imports)} required items imported successfully!")

print("\nDone!")
