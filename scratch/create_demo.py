import zipfile
import csv
import io
import shutil
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
demo_dir = project_root / "examples"
demo_dir.mkdir(exist_ok=True)
zip_path = demo_dir / "demo_vendor_pack.zip"

inductors = [
    ("POSM-DEMO-L-220nH", 220),
    ("POSM-DEMO-L-470nH", 470),
    ("POSM-DEMO-L-198.9nH", 198.94368),
    ("POSM-DEMO-L-474.1nH", 474.10245),
]

capacitors = [
    ("POSM-DEMO-C-636.62pF", 636.61977),
    ("POSM-DEMO-C-267.14pF", 267.13947),
]

tmp_dir = project_root / "scratch" / "tmp_demo"
tmp_dir.mkdir(exist_ok=True, parents=True)

with open(tmp_dir / "demo_inductors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Manufacturer", "Part Number", "Size", "Inductance", "Tolerance", "Irms", "Isat", "DCR Typ", "SRF Min", "Series"])
    for name, v in inductors:
        writer.writerow(["POSM-DEMO", name, "0603", f"{v}nH", "5%", "500mA", "600mA", "", "100GHz", ""])

with open(tmp_dir / "demo_capacitors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Manufacturer", "Part Number", "Size", "Capacitance", "Tolerance", "Voltage", "ESR Typ", "SRF Min", "Series"])
    for name, v in capacitors:
        writer.writerow(["POSM-DEMO", name, "0603", f"{v}pF", "5%", "50V", "", "100GHz", ""])

for name, _ in inductors + capacitors:
    with open(tmp_dir / f"{name}.s2p", "w") as f:
        f.write("! SYNTHETIC DEMO S2P\n")
        f.write("# HZ S MA R 50\n")
        f.write("1e6 0 0 1 0 1 0 0 0\n")
        f.write("100e6 0 0 1 0 1 0 0 0\n")

with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write(tmp_dir / "demo_inductors.csv", "demo_inductors.csv")
    zf.write(tmp_dir / "demo_capacitors.csv", "demo_capacitors.csv")
    for name, _ in inductors + capacitors:
        zf.write(tmp_dir / f"{name}.s2p", f"{name}.s2p")

shutil.rmtree(tmp_dir)
print(f"Created {zip_path}")
