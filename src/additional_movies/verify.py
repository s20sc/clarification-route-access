from pathlib import Path
from hashlib import sha256
import json
r=Path(__file__).resolve().parent
m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert sha256((r/n).read_bytes()).hexdigest()==h,n
print('4 original movie files and 2 wrapper files verified; no new coding performed')
