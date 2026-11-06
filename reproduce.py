"""Recalculate archived comparisons and check optional simulator recordings."""
import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "Quantitative_Data.zip": "quantitative",
    "Controller_Records.zip": "controllers",
    "Observation_Records.zip": "observations",
    "Additional_Original_Videos.zip": "additional_movies",
    "Display_Videos.zip": "display_videos",
}


def executable_tree(source):
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def extract(archive, destination, namespace):
    with zipfile.ZipFile(archive) as container:
        for item in container.infolist():
            member = Path(item.filename)
            if member.is_absolute() or ".." in member.parts or not member.parts or member.parts[0] != namespace:
                raise ValueError("Unexpected archive path: " + item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        container.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = {"run_id": uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "results.json").write_text(json.dumps({
        **run_identity, "status": "incomplete",
        "note": "Only status pass identifies a completed run. Other files in this directory may belong to an earlier run."
    }, indent=2) + "\n")
    if sys.flags.optimize:
        parser.error("Run with assertions enabled (omit -O).")
    hashes = json.loads((ROOT / "data-sha256.json").read_text())
    source_map = json.loads((ROOT / "source-map.json").read_text())
    available = {name: args.data_dir / name for name in GROUPS if (args.data_dir / name).is_file()}
    for required in ("Quantitative_Data.zip", "Controller_Records.zip"):
        if required not in available:
            parser.error("Missing required archive: " + required)
    for name, path in available.items():
        if sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Download digest mismatch: " + name)
    result = {**run_identity, "status": "pass", "archive_digests_verified": sorted(available), "source_files_verified": 0,
              "quantitative": "NOT RUN", "controllers": "NOT RUN",
              "observations": "NOT RUN: observation archive absent",
              "additional_movies": "NOT RUN: additional archive absent"}
    with tempfile.TemporaryDirectory(prefix="clarification-replay-") as work:
        work = Path(work)
        for name, archive in available.items():
            extract(archive, work, GROUPS[name])
        for source, entry in source_map.items():
            if entry["archive"] not in available:
                continue
            current = (ROOT / source).read_bytes()
            archived = (work / entry["member"]).read_bytes()
            if sha256(current).hexdigest() != entry["source_sha256"]:
                raise ValueError("Source digest mismatch: " + source)
            if sha256(archived).hexdigest() != entry["archive_sha256"]:
                raise ValueError("Archived source digest mismatch: " + entry["member"])
            if executable_tree(current) != executable_tree(archived):
                raise ValueError("Source differs from archived executable statements: " + source)
            result["source_files_verified"] += 1

        def run(namespace, script, extra=()):
            # Execute the GitHub entry point with its archived evidence location.
            # Nested analysis imports use the verified frozen source in the data.
            source = ROOT / "src" / namespace / script
            evidence = work / namespace / script
            runner = ("import pathlib,sys; source=pathlib.Path(sys.argv[1]); "
                      "evidence=pathlib.Path(sys.argv[2]); sys.argv=[str(evidence),*sys.argv[3:]]; "
                      "exec(compile(source.read_bytes(),str(evidence),'exec'),"
                      "{'__name__':'__main__','__file__':str(evidence)})")
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", runner, str(source), str(evidence), *map(str,extra)],
                                  cwd=work, capture_output=True)
            (args.output_dir / (namespace + ".stdout.txt")).write_bytes(proc.stdout)
            if proc.stderr:
                (args.output_dir / (namespace + ".stderr.txt")).write_bytes(proc.stderr)
            proc.check_returncode()
            return proc.stdout

        quantitative = run("quantitative", "recompute.py")
        if quantitative != (work / "quantitative/expected-results.json").read_bytes():
            raise ValueError("Quantitative replay differs from expected-results.json")
        result["quantitative"] = {"status": "pass", "output_sha256": sha256(quantitative).hexdigest()}
        run("controllers", "verify.py")
        result["controllers"] = "pass"
        if "Observation_Records.zip" in available:
            extra = []
            if "Additional_Original_Videos.zip" in available:
                extra += ["--additional-dir", work / "additional_movies"]
            if "Display_Videos.zip" in available:
                extra += ["--videos-dir", work / "display_videos"]
            result["observations"] = json.loads(run("observations", "verify.py", extra))
        if "Additional_Original_Videos.zip" in available:
            run("additional_movies", "verify.py")
            result["additional_movies"] = "pass"
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
