"""Replay cached inference and verify exact regenerated output bytes."""
import subprocess, sys, json, hashlib
from pathlib import Path
subprocess.run([sys.executable, 'reproduce/canonical_rebuild.py'], check=True)
expected=json.loads(Path('reproduce/expected_sha256.json').read_text())
for name, digest in expected.items():
    got=Path('outputs/canonical_rows')/name
    assert hashlib.sha256(got.read_bytes()).hexdigest()==digest, f'Output mismatch: {name}'
print('PASS: all canonical output hashes match the tested replay.')
with open('outputs/tables.txt', 'w') as out:
    subprocess.run([sys.executable, 'reproduce/canonical_tables.py'], stdout=out, check=True)
print(Path('outputs/tables.txt').read_text())
