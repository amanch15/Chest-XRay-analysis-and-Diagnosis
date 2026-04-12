
from PIL import Image
from pathlib import Path
project_root = Path(r'C:\Users\Emann\Desktop\MAJOR PROJECT')
txt_val = r'data\processed\00008426_002.png'
abs_path = str(project_root / txt_val)
print(abs_path)
print('Exists:', Path(abs_path).exists())
try:
    Image.open(abs_path)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()

