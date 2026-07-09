import gzip
import sys
from pathlib import Path
import ijson

path = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 80

with gzip.open(path, "rb") as f:
    for i, ev in enumerate(ijson.items(f, "traceEvents.item")):
        if i >= n:
            break
        name = ev.get("name", "")
        cat = ev.get("cat", "")
        ph = ev.get("ph", "")
        dur = ev.get("dur", "")
        args = ev.get("args", {})
        print("=" * 120)
        print("idx:", i)
        print("ph:", ph)
        print("cat:", cat)
        print("name:", str(name)[:300])
        print("dur:", dur)
        print("args keys:", list(args.keys())[:30] if isinstance(args, dict) else type(args))
        if isinstance(args, dict):
            for k in list(args.keys())[:15]:
                print(" ", repr(k), "=>", repr(args[k])[:300])
