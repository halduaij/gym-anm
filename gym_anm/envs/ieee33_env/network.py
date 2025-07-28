import numpy as np
import re

# Network data for the IEEE 33-bus distribution system
# extracted from the MATPOWER case33bw file.


def _load_case33bw(path):
    text = open(path).read()

    baseMVA = float(re.search(r"mpc.baseMVA\s*=\s*(\d+)", text).group(1))

    def parse_block(name):
        block = re.search(r"mpc.%s = \[(.*?)\];" % name, text, re.S).group(1)
        rows = []
        for line in block.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            line = line.split('%')[0].strip()
            if line.endswith(';'):
                line = line[:-1]
            nums = [float(n) for n in re.findall(r"-?\d+\.\d+|-?\d+", line)]
            if nums:
                rows.append(nums)
        return rows

    bus = parse_block("bus")
    branch = parse_block("branch")
    gen = parse_block("gen")
    return baseMVA, bus, branch, gen


# path of the original MATPOWER case file placed next to this script
CASE_PATH = __file__.replace("network.py", "case33bw.m")
baseMVA, bus_rows, branch_rows, _ = _load_case33bw(CASE_PATH)

# Base values for per-unit conversion
Vbase = bus_rows[0][9] * 1e3
Sbase = baseMVA * 1e6

# Bus specification array
bus = []
for row in bus_rows:
    bus_id = int(row[0] - 1)
    if row[1] == 3:
        bus_type = 0  # slack
    else:
        bus_type = 1  # PQ bus
    base_kv = row[9]
    bus.append([bus_id, bus_type, base_kv, 1.05, 0.95])

# Branch specification array
branch = []
for row in branch_rows:
    f = int(row[0] - 1)
    t = int(row[1] - 1)
    r_pu = row[2] / (Vbase ** 2 / Sbase)
    x_pu = row[3] / (Vbase ** 2 / Sbase)
    branch.append([f, t, r_pu, x_pu, 0.0, 0.0, 1, 0])

# Device specification array
# Slack generator
device = [
    [0, 0, 0, None, 999, -999, 999, -999, None, None, None, None, None, None, None]
]
# Loads at all other buses
dev_id = 1
for row in bus_rows[1:]:
    bus_id = int(row[0] - 1)
    Pd = row[2] / 1000.0  # kW to MW
    Qd = row[3] / 1000.0
    qp = Qd / Pd if Pd else 0.0
    device.append([dev_id, bus_id, -1, qp, 0, -Pd, None, None, None, None, None, None, None, None, None])
    dev_id += 1

# Add a couple of controllable capacitor banks for voltage regulation
device.append([dev_id, 8, 4, None, 0, 0, 1.0, -1.0, None, None, None, None, None, None, None])
dev_id += 1
device.append([dev_id, 25, 4, None, 0, 0, 1.0, -1.0, None, None, None, None, None, None, None])
dev_id += 1

# Add an OLTC on the slack branch between bus 0 and bus 1
device.append([dev_id, 0, 5, 1, 1.1, 0.9, None, None, None, None, None, None, None, None, None])
dev_id += 1

network = {"baseMVA": baseMVA}
network["bus"] = np.array(bus, dtype=float)
network["device"] = np.array(device, dtype=object)
network["branch"] = np.array(branch, dtype=float)
