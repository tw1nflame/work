python3 - <<'PY'
import zipfile
import glob

wheel = glob.glob('wheels/torch-2.7.1-*.whl')
if not wheel:
    raise SystemExit('wheel not found')
wheel = wheel[0]

with zipfile.ZipFile(wheel) as z:
    meta = [n for n in z.namelist() if n.endswith('METADATA')]
    if not meta:
        raise SystemExit('METADATA not found in wheel')
    text = z.read(meta[0]).decode('utf-8', errors='replace')

for line in text.splitlines():
    if line.startswith('Requires-Dist: '):
        print(line[len('Requires-Dist: '):])
PY
