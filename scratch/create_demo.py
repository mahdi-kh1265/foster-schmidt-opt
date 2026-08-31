import csv
import zipfile
from pathlib import Path
import os
import shutil

examples_dir = Path("examples")
examples_dir.mkdir(exist_ok=True)

coilcraft_headers = ["Manufacturer", "Part Number", "Inductance", "Tolerance", "Irms", "Isat", "DCR Typ", "SRF Min", "Size", "Series", "Validity Hz Lo", "Validity Hz Hi"]
murata_headers = ["Manufacturer", "Part Number", "Capacitance", "Tolerance", "Rated Voltage", "ESR", "Size", "Temperature Characteristic", "Validity Hz Lo", "Validity Hz Hi"]

inductors = []
# Values 1nH to 10uH
for v in [1, 2.2, 4.7, 10, 22, 47, 100, 220, 470, 1000, 2200, 4700, 10000]:
    inductors.append({
        "Manufacturer": "POSM-DEMO",
        "Part Number": f"POSM-DEMO-L-{v}nH",
        "Inductance": f"{v}e-9",
        "Tolerance": "0.10",
        "Irms": "1.5",
        "Isat": "2.0",
        "DCR Typ": "0.1",
        "SRF Min": "1000", # Fake SRF
        "Size": "0402",
        "Series": "DEMO-L", "Validity Hz Lo": "1000", "Validity Hz Hi": "10000000000"
    })

capacitors = []
# Values 1pF to 10nF
for v in [1, 2.2, 4.7, 10, 22, 47, 100, 220, 470, 1000, 2200, 4700, 10000]:
    capacitors.append({
        "Manufacturer": "POSM-DEMO",
        "Part Number": f"POSM-DEMO-C-{v}pF",
        "Capacitance": f"{v}e-12",
        "Tolerance": "0.05",
        "Rated Voltage": "50",
        "ESR": "0.1",
        "Size": "0402",
        "Temperature Characteristic": "C0G", "Validity Hz Lo": "1000", "Validity Hz Hi": "10000000000"
    })

tmp_dir = examples_dir / "demo_tmp"
tmp_dir.mkdir(exist_ok=True)

coilcraft_dir = tmp_dir / "coilcraft"
coilcraft_dir.mkdir(exist_ok=True)
with open(coilcraft_dir / "demo_inductors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=coilcraft_headers)
    w.writeheader()
    for r in inductors: w.writerow(r)

murata_dir = tmp_dir / "murata"
murata_dir.mkdir(exist_ok=True)
with open(murata_dir / "demo_capacitors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=murata_headers)
    w.writeheader()
    for r in capacitors: w.writerow(r)

for r in inductors:
    s2p = coilcraft_dir / f"{r['Part Number']}.s2p"
    with open(s2p, "w") as f:
        f.write("! SYNTHETIC DEMO S2P\n# HZ S MA R 50\n1e6 0 0 1 0 1 0 0 0\n100e6 0 0 1 0 1 0 0 0\n")

for r in capacitors:
    s2p = murata_dir / f"{r['Part Number']}.s2p"
    with open(s2p, "w") as f:
        f.write("! SYNTHETIC DEMO S2P\n# HZ S MA R 50\n1e6 0 0 1 0 1 0 0 0\n100e6 0 0 1 0 1 0 0 0\n")

zip_path = examples_dir / "demo_vendor_pack.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            p = Path(root) / file
            arcname = p.relative_to(tmp_dir)
            zf.write(p, arcname)

shutil.rmtree(tmp_dir)
