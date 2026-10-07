import csv, sys, numpy as np
from pathlib import Path
for n in sys.argv[2:]:
    d = sorted(p for p in Path("runs", n).iterdir() if (p / "fitness.csv").exists())[-1]
    rows = list(csv.DictReader(open(d / "fitness.csv"))); w = max(1, len(rows) // 10)
    r = [float(x[sys.argv[1]]) for x in rows]
    print(f"{n:22s} {sys.argv[1]:9s}", " ".join(f"{np.mean(r[i:i+w]):5.0f}" for i in range(0, len(r), w)))
