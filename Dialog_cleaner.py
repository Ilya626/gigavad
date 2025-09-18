import json, sys, os

in_path = "Dnd session records, 01_07.txt"
out_path = "Dnd session records, 01_07_output.txt"

with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        name = os.path.basename(obj["audio"])
        if name.endswith(".wav"):
            name = name[:-4]
        text = json.dumps(obj["text"], ensure_ascii=False)
        fout.write(f"{name}: {text}\n")
