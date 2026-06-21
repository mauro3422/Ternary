"""
Guarda una version del modelo entrenado con su configuracion y metadatos.
Uso: python save_model.py v2_hgrn_80m "HGRN 80M Shakespeare"
"""

import os
import sys
import shutil
import json
from datetime import datetime

def save_version(version_name, description=""):
    base = os.path.dirname(__file__)
    version_dir = os.path.join(base, 'models', version_name)
    os.makedirs(version_dir, exist_ok=True)

    ckpt_src = os.path.join(base, 'checkpoint.pt')
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, os.path.join(version_dir, 'checkpoint.pt'))
        print(f"  checkpoint.pt -> {version_dir}/")

    bin_src = os.path.join(base, 'engine', 'model.bin')
    if os.path.exists(bin_src):
        shutil.copy2(bin_src, os.path.join(version_dir, 'model.bin'))
        print(f"  model.bin -> {version_dir}/")

    meta = {
        'version': version_name,
        'description': description,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'files': {
            'checkpoint': os.path.exists(ckpt_src),
            'model_bin': os.path.exists(bin_src),
        }
    }
    with open(os.path.join(version_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json -> {version_dir}/")

    size = sum(os.path.getsize(os.path.join(dirpath, f))
               for dirpath, _, files in os.walk(version_dir)
               for f in files)
    print(f"\nVersion '{version_name}' guardada ({size / 1024:.1f} KB)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python save_model.py <version_name> [descripcion]")
        sys.exit(1)
    save_version(sys.argv[1], ' '.join(sys.argv[2:]))
