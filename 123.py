python -m pip install \
  --dry-run \
  --ignore-installed \
  --report torch271-report.json \
  torch==2.7.1

python - <<'PY'
import json
with open("torch271-report.json") as f:
    data = json.load(f)

for item in data.get("install", []):
    meta = item.get("metadata", {})
    name = meta.get("name")
    version = meta.get("version")
    if name and version:
        print(f"{name}=={version}")
PY
