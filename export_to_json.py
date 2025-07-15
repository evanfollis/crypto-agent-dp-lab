#!/usr/bin/env python3
import sys, os, json

def export_to_json(root_dir, out_file):
    data = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_dir)
            try:
                with open(full, encoding='utf-8') as f:
                    data[rel] = f.read()
            except Exception as e:
                print(f"⚠️ Skipping {rel}: {e}", file=sys.stderr)
    with open(out_file, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(data)} entries to {out_file!r}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: script.py <root_directory>", file=sys.stderr)
        sys.exit(1)
    root = sys.argv[1]
    name = os.path.basename(os.path.abspath(root)) or 'export'
    export_to_json(root, f"{name}.json")
