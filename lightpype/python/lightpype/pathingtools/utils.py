"""
Project path resolver for 3DFiberSpectrograph.

Automatically discovers the project root based on fixed directory structure.
This file is always located at:
    <project_root>/python/lightpype/pathingtools/utils.py
So the project root is exactly 3 levels up.
"""

import sys
from pathlib import Path

# Project root is ALWAYS 3 levels above this file (because we're in .../python/lightpype/pathingtools/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Verify that the structure is as expected
if not (PROJECT_ROOT / "python").is_dir():
    raise RuntimeError(
        f"Expected project root at {PROJECT_ROOT}, but 'python/' directory not found.\n"
        f"Make sure this file is located at:\n"
        f"  {PROJECT_ROOT}/python/lightpype/pathingtools/utils.py\n"
        f"Current file: {Path(__file__).resolve()}"
    )

def get_config_path(*parts: str) -> Path:
    """Return full path to a config file under project_root/python/lightpype/config/"""
    return PROJECT_ROOT / "python" / "lightpype" / "config" / Path(*parts)

def get_gpio_config_path() -> Path:
    """Return path to gpio_control.json"""
    return PROJECT_ROOT / "python" / "lightpype" / "gpio_control" / "gpio_control.json"

def get_spectrometer_config_path() -> Path:
    """Return path to spectrometer_config.json"""
    return PROJECT_ROOT / "python" / "lightpype" / "spectrometer" / "spectrometer_config.json"

def get_data_path(*parts: str) -> Path:
    """Return path to data directory under project_root/python/lightpype/data/"""
    return PROJECT_ROOT / "python" / "lightpype" / "data" / Path(*parts)

def get_docs_path(*parts: str) -> Path:
    """Return path to documentation files under project_root/python/doc/"""
    return PROJECT_ROOT / "python" / "doc" / Path(*parts)

# Add python/ to Python path so we can import from lightpype.* anywhere
def add_lightpype_to_path():
    """Add the python/ directory to sys.path so imports like 'from lightpype.gpio_control import LEDManager' work."""
    python_dir = PROJECT_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

# Call this once at module import to make imports work everywhere
add_lightpype_to_path()


# --- Self-Verification Tests ---
if __name__ == "__main__":
    """
    Run this file directly to verify all critical paths exist.
    Usage: python lightpype/pathingtools/utils.py
    """
    import sys

    print("🔍 Verifying project paths...")

    # Test 1: Project root is correctly located
    print(f"✅ Project root (determined by path structure): {PROJECT_ROOT}")
    assert (PROJECT_ROOT / "python").is_dir(), "'python/' directory not found at project root"
    print("✅ python/ directory exists")

    # Test 2: GPIO config
    gpio_config = get_gpio_config_path()
    assert gpio_config.exists(), f"GPIO config file not found at {gpio_config}"
    print(f"✅ GPIO config: {gpio_config}")

    # Test 3: Spectrometer config
    spectrometer_config = get_spectrometer_config_path()
    assert spectrometer_config.exists(), f"Spectrometer config file not found at {spectrometer_config}"
    print(f"✅ Spectrometer config: {spectrometer_config}")

    # Test 4: Docs directory
    docs_dir = get_docs_path()
    assert docs_dir.exists(), f"Documentation directory not found at {docs_dir}"
    print(f"✅ Docs directory: {docs_dir}")

    # Test 5: Python path was added
    python_dir = PROJECT_ROOT / "python"
    assert str(python_dir) in sys.path, f"'{python_dir}' not added to sys.path"
    print(f"✅ Python path added: {python_dir}")

    # Test 6: Import sanity check
    try:
        from lightpype.gpio_control.led_manager import LEDManager
        print("✅ Can import LEDManager from lightpype.gpio_control")
    except ImportError as e:
        print(f"❌ Failed to import LEDManager: {e}")
        sys.exit(1)

    # Test 7: get_config_path() with multiple parts
    test_path = get_config_path("test", "subdir", "config.json")
    print(f"✅ get_config_path() supports multiple parts: {test_path}")

    print("\n🎉 All path checks passed! Project is ready to run.")