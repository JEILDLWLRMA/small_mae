#!/usr/bin/env python3
"""
Fix timm 0.3.2 compatibility with PyTorch >= 1.9.0
This script patches the torch._six import issue in timm.
"""

import os
import sys
import shutil

def fix_timm_compatibility():
    """Patch timm to work with PyTorch >= 1.9.0"""
    try:
        import timm
        timm_path = os.path.dirname(timm.__file__)
        helpers_path = os.path.join(timm_path, 'models', 'layers', 'helpers.py')
        
        if not os.path.exists(helpers_path):
            print(f"Error: {helpers_path} not found")
            print(f"timm path: {timm_path}")
            return False
        
        # Read the file
        with open(helpers_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'from collections.abc' in content and 'from torch._six import container_abcs' not in content:
            print("timm compatibility already fixed")
            return True
        
        # Backup original file
        backup_path = helpers_path + '.backup'
        if not os.path.exists(backup_path):
            shutil.copy2(helpers_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Apply patch: replace torch._six.container_abcs with collections.abc
        if 'from torch._six import container_abcs' in content:
            # Simple fix: replace with collections.abc
            # Note: container_abcs was used for isinstance checks, we'll use Iterable
            content = content.replace(
                'from torch._six import container_abcs',
                'try:\n    from collections.abc import Iterable\n    container_abcs = Iterable\nexcept ImportError:\n    from collections import Iterable\n    container_abcs = Iterable'
            )
            
            # Also check for any usage and replace if needed
            # container_abcs is typically used as: isinstance(x, container_abcs)
            # We can keep it as Iterable which works the same way
        
        # Write back
        with open(helpers_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {helpers_path}")
        print("timm should now work with PyTorch >= 1.9.0")
        print("\nTo restore original file, use:")
        print(f"  cp {backup_path} {helpers_path}")
        return True
        
    except ImportError as e:
        print("=" * 60)
        print("ERROR: timm is not installed yet.")
        print("=" * 60)
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        print("\nThen run this patch script again:")
        print("  python scripts/fix_timm_compatibility.py")
        print("\nOr install timm directly:")
        print("  pip install timm==0.3.2")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"Error patching timm: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = fix_timm_compatibility()
    sys.exit(0 if success else 1)

