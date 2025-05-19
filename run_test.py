import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from test_model import test_property_model

if __name__ == "__main__":
    success = test_property_model()
    if success:
        print("All tests passed successfully!")
    else:
        print("Tests failed!") 