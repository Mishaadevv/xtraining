"""Backend tests.

Runs on a bare Python interpreter — no pytest, no torch:

    python tests/test_backend.py

Anything that would need the ML runtime is skipped, not faked.
"""

from __future__ import annotations

import importlib.util as importlib_util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from zeqouxtraining import config as config_mod  # noqa: E402
from zeqouxtraining import datasets, deps, estimator, events, hardware  # noqa: E402
from zeqouxtraining.checkpoints import list_checkpoints, prune  # noqa: E402
from zeqouxtraining.errors import humanize  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"  FAIL  {name}  {detail}")


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

def test_events() -> None:
    section("Event protocol")

    buffer = io.StringIO()
    real_stream = events._EVENT_STREAM
    events._EVENT_STREAM = buffer  # type: ignore[assignment]
    try:
        events.emit("training-progress", {"step": 3, "loss": 1.25})
        events.log("hello", level="warn")
        events.fail("nope", hint="do this", code="x")
    finally:
        events._EVENT_STREAM = real_stream  # type: ignore[assignment]

    lines = [json.loads(line) for line in buffer.getvalue().strip().splitlines()]
    check("emit writes one JSON object per line", len(lines) == 3)
    check("event name is preserved", lines[0]["event"] == "training-progress")
    check("detail is preserved", lines[0]["detail"]["step"] == 3)
    check("log carries a level", lines[1]["detail"]["level"] == "warn")
    check("fail carries hint and code",
          lines[2]["detail"]["hint"] == "do this" and lines[2]["detail"]["code"] == "x")


# --------------------------------------------------------------------------- #
# Dependencies and hardware
# --------------------------------------------------------------------------- #

def test_deps_and_hardware() -> None:
    section("Dependencies and hardware")
    info = deps.inspect()
    check("python version reported", bool(info["python"]["version"]))
    check("package table is populated", len(info["packages"]) >= 8)
    check("missing_core is a list", isinstance(info["missing_core"], list))
    check("capabilities include qlora", "qlora" in info["capabilities"])
    check("qlora capability requires bitsandbytes",
          "bitsandbytes" in info["capabilities"]["qlora"]["requires"])

    plan = deps.install_plan("cu124")
    check("install plan numbers the cuda wheel index",
          "download.pytorch.org/whl/cu124" in plan["command"])
    check("install plan is a real argv", plan["argv"][1:4] == ["-m", "pip", "install"])

    detected = hardware.detect()
    check("hardware reports an os", bool(detected["os"]["system"]))
    check("hardware reports cpu cores", bool(detected["cpu"]["logical_cores"]))
    check("gpu probe is boolean", isinstance(detected["gpu"]["available"], bool))
    check("cuda probe is boolean", isinstance(detected["cuda_ready"], bool))
    check("training device is cuda or cpu", detected["training_device"] in ("cuda", "cpu"))
    if not detected["gpu"]["available"]:
        # This machine genuinely has no NVIDIA GPU; the app must say why.
        check("a reason is given when no GPU is found", bool(detected["gpu"].get("reason")))
        check("blockers explain the missing GPU", len(detected["cuda_blockers"]) >= 1)
    if not detected["cuda_ready"]:
        check("training falls back to cpu honestly", detected["training_device"] == "cpu")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

def test_format_detection() -> None:
    section("Dataset format detection")
    cases = {
        "good.jsonl": "jsonl",
        "chat.json": "json",
        "qa.csv": "csv",
        "notes.txt": "txt",
        "messy.jsonl": "jsonl",
    }
    for name, expected in cases.items():
        actual = datasets.detect_format(str(FIXTURES / name))
        check(f"{name} -> {expected}", actual == expected, f"got {actual}")


def test_good_dataset() -> None:
    section("Clean dataset")
    report = datasets.validate(str(FIXTURES / "good.jsonl"), context_length=512)
    check("clean dataset has no errors", report["status"] in ("ok", "warnings"), report["status"])
    check("clean dataset is ok", report["ok"] is True)
    check("record count is read", report["dataset"]["records"] == 6, str(report["dataset"]))
    check("instruction/output pair is detected",
          report["mapping"]["kind"] == "pair", json.dumps(report["mapping"]))
    check("the correct fields are picked",
          report["mapping"].get("instruction_field") == "instruction"
          and report["mapping"].get("output_field") == "output")
    check("usable equals record count", report["stats"]["usable"] == 6, str(report["stats"]["usable"]))
    check("no duplicates in the clean file", report["stats"]["duplicates"] == 0)
    check("preview is produced", len(report["preview"]) > 0)
    check("preview keeps the instruction text",
          "Summarise" in report["preview"][0], report["preview"][0][:80])


def test_translations() -> None:
    section("Unicode and non-English content")
    report = datasets.validate(str(FIXTURES / "good.jsonl"), context_length=512)
    joined = "\n".join(report["preview"])
    check("cyrillic survives validation", "Обучение" in joined)


def test_messy_dataset() -> None:
    section("Messy dataset")
    report = datasets.validate(str(FIXTURES / "messy.jsonl"), context_length=64)
    codes = {issue["code"] for issue in report["issues"]}
    check("duplicates are detected", "duplicates" in codes, str(codes))
    check("unusable records are detected", "unusable_records" in codes, str(codes))
    check("very short samples are detected", "very_short" in codes, str(codes))
    check("over-length samples are detected", "over_length" in codes, str(codes))
    check("status is at least warnings", report["status"] in ("warnings", "errors"))
    dup = report["stats"]["duplicates"]
    check("exactly one duplicate is counted", dup == 1, f"got {dup}")
    empty = report["stats"]["empty"]
    check("both unusable records are counted", empty == 2, f"got {empty}")
    reasons = report["stats"]["empty_reasons"]
    check("unusable reasons are itemised",
          reasons.get("empty_output") == 2, str(reasons))
    short = report["stats"]["short"]
    check("the one-letter target is flagged short", short == 1, f"got {short}")
    over = report["stats"]["over_length"]
    check("one sample exceeds the context", over == 1, f"got {over}")


def test_broken_dataset() -> None:
    section("Broken dataset")
    report = datasets.validate(str(FIXTURES / "broken.json"))
    check("broken json fails validation", report["ok"] is False)
    check("status is errors", report["status"] == "errors")
    check("an error message is present", bool(report["error"]))
    message = (report["error"] or {}).get("message", "")
    check("the message names the line", "line" in message.lower(), message)
    check("a hint accompanies the error", bool((report["error"] or {}).get("hint")))
    # A failed validation must still describe the source, or the caller loses
    # the format it already knew about.
    check("the report still names the format", report["dataset"].get("format") == "json",
          json.dumps(report["dataset"]))
    check("the failed report has the same dataset keys as a successful one",
          {"path", "name", "format", "bytes", "records", "files"} <= set(report["dataset"]),
          json.dumps(sorted(report["dataset"])))


def test_missing_file() -> None:
    section("Missing file")
    report = datasets.validate(str(FIXTURES / "does-not-exist.json"))
    check("missing file fails validation", report["ok"] is False)
    check("missing file code is not_found",
          (report["error"] or {}).get("code") == "not_found", json.dumps(report["error"]))


def test_chat_dataset() -> None:
    section("Chat dataset")
    report = datasets.validate(str(FIXTURES / "chat.json"), context_length=512)
    check("chat kind detected", report["mapping"]["kind"] == "chat", json.dumps(report["mapping"]))
    check("messages field is used", report["mapping"].get("messages_field") == "messages")
    check("container key is reported", report["dataset"]["container_key"] == "data")
    check("all three conversations are usable", report["stats"]["usable"] == 3, str(report["stats"]))
    check("roles are counted",
          report["stats"]["roles"].get("assistant") == 3, str(report["stats"]["roles"]))

    records, _meta = datasets.load_records(str(FIXTURES / "chat.json"))
    samples = datasets.normalize(records, report["mapping"])
    check("chat samples keep the messages structure", "messages" in samples[0])
    check("sharegpt 'from/value' is converted to roles",
          samples[2]["messages"][1]["role"] == "assistant" and
          "checkpoint" in samples[2]["messages"][1]["content"])


def test_csv_and_txt() -> None:
    section("CSV and TXT")
    csv_report = datasets.validate(str(FIXTURES / "qa.csv"), context_length=512)
    check("csv produces samples", csv_report["stats"]["usable"] == 4, str(csv_report["stats"]))
    check("csv question/answer is detected",
          csv_report["mapping"]["kind"] == "pair", json.dumps(csv_report["mapping"]))

    txt_report = datasets.validate(str(FIXTURES / "notes.txt"), context_length=512)
    check("txt is split into paragraphs", txt_report["stats"]["records"] == 4,
          str(txt_report["stats"]))
    check("txt kind is text", txt_report["mapping"]["kind"] == "text")
    check("txt samples keep their content",
          "Gradient accumulation" in txt_report["preview"][2], txt_report["preview"][2][:80])


def test_folder_dataset() -> None:
    section("Folder dataset (homogeneous shards)")
    report = datasets.validate(str(FIXTURES / "folder_clean"), context_length=512)
    check("all shards are read", len(report["dataset"]["files"]) == 2,
          str(report["dataset"]["files"]))
    check("shards are concatenated", report["dataset"]["records"] == 12,
          str(report["dataset"]["records"]))
    check("every shard row is usable", report["stats"]["usable"] == 12,
          str(report["stats"]["usable"]))
    check("a homogeneous folder is clean", report["ok"] is True, report["status"])
    check("a homogeneous folder raises no shape warning",
          "mixed_folder" not in {i["code"] for i in report["issues"]})

    section("Folder dataset (broken and mixed files)")
    mixed = datasets.validate(str(FIXTURES), context_length=512)
    errors = mixed["dataset"].get("file_errors") or []
    check("the broken file is named rather than aborting the folder",
          any(e["file"] == "broken.json" for e in errors), str(errors))
    check("the good files were still read", mixed["dataset"]["records"] > 10,
          str(mixed["dataset"]["records"]))
    check("mixing record shapes is reported",
          "mixed_folder" in {i["code"] for i in mixed["issues"]},
          str([i["code"] for i in mixed["issues"]]))
    check("the mixed folder is not training-ready", mixed["ok"] is False)


def test_huggingface_source() -> None:
    section("Hugging Face dataset source")

    # The 'datasets' package is optional and not installed here, so the loader is
    # stubbed: what is under test is the plumbing around it, not Hugging Face.
    stub_rows = [
        {"instruction": f"question {index}", "output": f"answer {index}" * 4}
        for index in range(4)
    ]
    original = datasets.load_hf_dataset
    calls: list[tuple[str, str]] = []

    def fake_load(dataset_id: str, split: str = "train", max_records: int = 50_000):
        calls.append((dataset_id, split))
        return stub_rows

    try:
        datasets.load_hf_dataset = fake_load  # type: ignore[assignment]

        records, meta = datasets.resolve_source(hf_id="org/data", split="validation")
        check("a Hub id is routed to the Hugging Face loader", calls == [("org/data", "validation")], str(calls))
        check("Hub metadata is marked as remote", meta["source"] == "huggingface")
        check("Hub metadata has no local byte count", meta["bytes"] is None)
        check("the split is preserved in metadata", meta["split"] == "validation")

        report = datasets.validate(hf_id="org/data", split="validation", context_length=512)
        check("a Hub dataset validates into the same report shape",
              report["status"] in ("ok", "warnings") and report["ok"] is True, report["status"])
        check("the report says where the data came from",
              report["dataset"]["source"] == "huggingface", json.dumps(report["dataset"])[:120])
        check("the report reports the Hub format", report["dataset"]["format"] == "hf")
        check("the report does not pretend to know a byte size", report["dataset"]["bytes"] is None)
        check("Hub records are counted", report["dataset"]["records"] == 4, str(report["dataset"]))
        check("Hub rows are normalised and previewed", len(report["preview"]) == 4, str(len(report["preview"])))
        check("the pair mapping is detected for Hub rows", report["mapping"]["kind"] == "pair")
    finally:
        datasets.load_hf_dataset = original  # type: ignore[assignment]

    # Without the optional package the real loader must say so, not raise a
    # traceback or silently return nothing.
    import importlib.util

    if importlib.util.find_spec("datasets") is None:
        try:
            datasets.load_hf_dataset("org/data")
            honest = False
            detail = "no error raised"
        except datasets.DatasetError as exc:
            honest = exc.code == "missing_dependency" and "pip install datasets" in exc.hint
            detail = f"{exc.code}: {exc.message}"
        check("a missing 'datasets' package is reported with an install hint", honest, detail)

    empty = datasets.validate()
    check("validation with no source fails with a clear code",
          empty["ok"] is False and empty["issues"][0]["code"] == "no_source",
          json.dumps(empty["issues"])[:120])
    check("the no-source error is actionable", bool(empty["issues"][0].get("hint")))


def test_format_table() -> None:
    """The documented extension table is the contract the app offers to the user."""
    section("Format table")
    expected = {
        "data.json": "json",
        "data.jsonl": "jsonl",
        "data.ndjson": "jsonl",
        "data.jsonlines": "jsonl",
        "data.csv": "csv",
        "data.psv": "csv",
        "data.tsv": "tsv",
        "data.tab": "tsv",
        "data.txt": "txt",
        "data.text": "txt",
        "data.md": "txt",
        "data.markdown": "txt",
        "data.parquet": "parquet",
        "data.pq": "parquet",
        "data.arrow": "arrow",
        "data.feather": "arrow",
        "data.orc": "orc",
        "data.sqlite": "sqlite",
        "data.sqlite3": "sqlite",
        "data.db": "sqlite",
        "data.xlsx": "excel",
        "data.xlsm": "excel",
        "data.xls": "excel",
        "data.yaml": "yaml",
        "data.yml": "yaml",
    }
    for name, want in expected.items():
        got = datasets.detect_format(name)
        check(f"{name} -> {want}", got == want, f"got {got}")

    for name in ("data.jsonl.gz", "shard.ndjson.bz2", "part.csv.xz"):
        inner = name.split(".")[1]
        got = datasets.detect_format(name)
        check(f"{name} is read as {inner}", got == datasets.detect_format(f"x.{inner}"), f"got {got}")
        check(f"{name} reports its compression", datasets.detect_compression(name) is not None)

    try:
        datasets.detect_format("data.docx")
        unsupported = False
    except datasets.DatasetError as exc:
        unsupported = exc.code == "unsupported_format" and ".json" in exc.hint
    check("an unsupported extension names the supported ones", unsupported)

    try:
        datasets.detect_format("archive.gz")
        ambiguous = False
    except datasets.DatasetError as exc:
        ambiguous = exc.code == "unsupported_format" and "inner extension" in exc.hint
    check("a compressed file without an inner extension says how to name it", ambiguous)

    info = datasets.supported_formats()
    check("every format id is described", len(info["formats"]) == len(datasets.FORMAT_INFO))
    check("extensions are reported for the UI", ".parquet" in info["extensions"], str(info["extensions"]))
    check("compression codecs are reported", ".gz" in info["compression"], str(info["compression"]))
    check("export formats are reported", "jsonl" in info["export_formats"])
    check("availability is probed, not assumed",
          all(isinstance(f["available"], bool) for f in info["formats"]))
    check("an unavailable reader carries an install hint",
          all(f["hint"] or f["available"] for f in info["formats"]),
          str([f for f in info["formats"] if not f["available"]][:2]))
    check("stdlib formats are always available",
          all(f["available"] for f in info["formats"] if f["requires"] is None))


def test_extra_formats() -> None:
    """Formats that need real files are exercised from a throwaway folder."""
    section("Additional formats (tsv, gzip, markdown, sqlite)")
    import gzip
    import sqlite3

    rows = [{"instruction": f"question {i}", "output": f"answer {i}"} for i in range(6)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        (root / "pairs.tsv").write_text(
            "instruction\toutput\n" + "\n".join(f"{r['instruction']}\t{r['output']}" for r in rows),
            encoding="utf-8",
        )
        tsv = datasets.validate(str(root / "pairs.tsv"), context_length=512)
        check("a .tsv file is read as tab separated",
              tsv["dataset"]["records"] == 6 and tsv["dataset"]["format"] == "tsv",
              json.dumps(tsv["dataset"])[:120])
        check("tsv columns are mapped like csv", tsv["mapping"]["kind"] == "pair",
              json.dumps(tsv["mapping"]))

        (root / "shard-01.jsonl.gz").write_bytes(gzip.compress(
            "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        ))
        gz = datasets.validate(str(root / "shard-01.jsonl.gz"), context_length=512)
        check("a gzipped JSONL file is decompressed transparently",
              gz["dataset"]["records"] == 6, str(gz["dataset"]["records"]))
        check("the report names the compression", gz["dataset"]["compression"] == "gzip")
        check("the inner format is reported", gz["dataset"]["format"] == "jsonl")

        gold = datasets.export_dataset(str(root / "shard-01.jsonl.gz"), output=str(root / "plain.jsonl"))
        check("a compressed source exports to a plain single file",
              gold["format"] == "jsonl" and Path(gold["output"]).exists(), json.dumps(gold)[:120])

        (root / "broken.jsonl.gz").write_bytes(b"this is not gzip at all")
        broken = datasets.validate(str(root / "broken.jsonl.gz"))
        check("a corrupt gzip file is reported as such",
              broken["issues"][0]["code"] == "bad_compression", broken["issues"][0]["message"])

        (root / "notes.md").write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
        md = datasets.validate(str(root / "notes.md"), context_length=512)
        check("markdown is read as text", md["dataset"]["format"] == "txt" and md["stats"]["records"] == 2,
              json.dumps(md["dataset"])[:120])

        connection = sqlite3.connect(root / "pairs.db")
        connection.execute("CREATE TABLE pairs (instruction TEXT, output TEXT)")
        connection.execute("CREATE TABLE small (x INTEGER)")
        connection.executemany("INSERT INTO pairs VALUES (?, ?)",
                               [(r["instruction"], r["output"]) for r in rows])
        connection.execute("INSERT INTO small VALUES (1)")
        connection.commit()
        connection.close()

        db = datasets.validate(str(root / "pairs.db"), context_length=512)
        check("a sqlite database is read without extra packages",
              db["dataset"]["records"] == 6, json.dumps(db["dataset"])[:160])
        check("the largest table is chosen", db["dataset"]["items"] == "pairs",
              str(db["dataset"].get("items")))
        check("sqlite columns map like any table", db["mapping"]["kind"] == "pair")

        empty_db = root / "empty.db"
        sqlite3.connect(empty_db).close()
        empty = datasets.validate(str(empty_db))
        check("a database with no tables is a clean error", empty["issues"][0]["code"] == "empty",
              str(empty["issues"][0]))

        (root / "pairs.yaml").write_text(
            "\n".join(
                f"- instruction: {r['instruction']}\n  output: {r['output']}" for r in rows
            ),
            encoding="utf-8",
        )
        yml = datasets.validate(str(root / "pairs.yaml"), context_length=512)
        if importlib_util.find_spec("yaml") is not None:
            check("yaml records are read", yml["dataset"]["records"] == 6, str(yml["dataset"]["records"]))
        else:
            yml_issue = yml["issues"][0]
            check("yaml without pyyaml is an honest missing dependency",
                  yml_issue["code"] == "missing_dependency" and "pip install pyyaml" in yml_issue["hint"],
                  json.dumps(yml_issue)[:110])


def test_single_file_export() -> None:
    section("Single-file export")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source_records, _source_meta = datasets.load_records(str(FIXTURES / "good.jsonl"))
        source_texts = [
            sample["text"] for sample in datasets.normalize(
                source_records, datasets.detect_mapping(source_records)
            )
        ]

        for fmt in ("jsonl", "json", "csv", "tsv", "txt"):
            workdir = root / fmt
            workdir.mkdir()
            target = workdir / f"out.{fmt}"
            result = datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(target))
            written = sorted(p.name for p in workdir.iterdir())
            check(f"{fmt}: exactly one file is written",
                  written == [target.name] and result["output"] == str(target), str(written))
            check(f"{fmt}: the export reports the format it wrote", result["format"] == fmt, result["format"])
            check(f"{fmt}: every source record is exported", result["records"] == 6, str(result["records"]))
            reloaded = datasets.validate(str(target), context_length=512)
            if fmt == "txt":
                # A blank line is the only record boundary plain text has, so the
                # record count can differ. No content may be lost, though.
                content = target.read_text(encoding="utf-8")
                check("txt: every sample survives the export",
                      all(text in content for text in source_texts),
                      f"{len(source_texts)} samples written")
            else:
                check(f"{fmt}: the exported file reads back as a dataset",
                      reloaded["dataset"]["records"] == 6, json.dumps(reloaded["dataset"])[:120])
            check(f"{fmt}: the export is readable as {fmt}",
                  reloaded["dataset"]["format"] == fmt, reloaded["dataset"]["format"])

        # A folder of shards becomes one file, which is the whole point.
        merged = datasets.export_dataset(str(FIXTURES / "folder_clean"), output=str(root / "merged.jsonl"))
        check("a shard folder exports to a single file",
              merged["records"] == 12 and merged["source"]["files"], json.dumps(merged)[:140])
        check("the export keeps the mapping it used", merged["mapping"]["kind"] == "pair")
        check("exports are normalised by default", merged["mode"] == "normalised")

        raw = datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(root / "raw.jsonl"), raw=True)
        check("raw mode keeps the original fields", "instruction" in raw["fields"], str(raw["fields"]))

        try:
            datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(root / "merged.jsonl"))
            refused = False
        except datasets.DatasetError as exc:
            refused = exc.code == "exists"
        check("exporting over an existing file is refused, not silently merged", refused)

        again = datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(root / "merged.jsonl"),
                                        overwrite=True)
        check("overwriting is possible when explicitly allowed",
              again["records"] == 6, json.dumps(again)[:100])

        try:
            datasets.export_dataset(str(FIXTURES / "good.jsonl"), output="")
            no_output = False
        except datasets.DatasetError as exc:
            no_output = exc.code == "no_output"
        check("exporting without a destination is refused", no_output)

        try:
            datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(root / "out.docx"))
            bad_format = False
        except datasets.DatasetError as exc:
            bad_format = exc.code == "unsupported_format" and "jsonl" in exc.hint
        check("an unsupported export format lists the supported ones", bad_format)

        if importlib_util.find_spec("pyarrow") is None:
            try:
                datasets.export_dataset(str(FIXTURES / "good.jsonl"), output=str(root / "out.parquet"))
                parquet = None
            except datasets.DatasetError as exc:
                parquet = exc
            check("parquet without pyarrow is refused with an install hint",
                  parquet is not None and parquet.code == "missing_dependency"
                  and "pip install pyarrow" in parquet.hint,
                  parquet.message if parquet else "no error")
            check("the refused parquet export left no file behind",
                  not (root / "out.parquet").exists())


def test_mapping_override() -> None:
    section("Manual mapping override")
    report = datasets.validate(
        str(FIXTURES / "good.jsonl"),
        mapping={"kind": "text", "text_field": "output"},
        context_length=512,
    )
    check("an explicit mapping is honoured", report["mapping"]["kind"] == "text")
    check("the explicit field is used",
          "training run finished" not in report["preview"][0],
          report["preview"][0][:60])


# --------------------------------------------------------------------------- #
# Configuration and estimation
# --------------------------------------------------------------------------- #

def test_config_normalisation() -> None:
    section("Configuration")
    cfg = config_mod.normalize({"method": "QLORA", "epochs": "4", "batch_size": None,
                                "learning_rate": "0.0001"})
    check("method is lower-cased", cfg["method"] == "qlora")
    check("string numbers are coerced", cfg["epochs"] == 4)
    check("invalid numbers fall back to defaults", cfg["batch_size"] == config_mod.DEFAULTS["batch_size"])
    check("learning rate string is parsed", abs(cfg["learning_rate"] - 1e-4) < 1e-12)
    check("qlora forces 4bit quantization", cfg["quantization"] == "4bit")
    check("defaults are complete", all(k in cfg for k in config_mod.DEFAULTS))
    check("effective batch size multiplies", config_mod.effective_batch_size(
        config_mod.normalize({"batch_size": 3, "gradient_accumulation": 4})) == 12)


def test_config_validation() -> None:
    section("Configuration validation")
    empty = config_mod.validate(config_mod.normalize({}), {"cuda_ready": False})
    codes = {issue["code"] for issue in empty}
    check("missing base model is an error", "no_base_model" in codes, str(codes))
    check("missing dataset is an error", "no_dataset" in codes, str(codes))

    qlora_cpu = config_mod.validate(
        config_mod.normalize({"base_model": "x", "method": "qlora",
                              "dataset": {"path": "y"}}),
        {"cuda_ready": False},
    )
    check("qlora without cuda is blocked",
          any(i["code"] == "qlora_requires_cuda" and i["severity"] == "error" for i in qlora_cpu))

    bad_lr = config_mod.validate(
        config_mod.normalize({"base_model": "x", "learning_rate": 0.5,
                              "dataset": {"path": "y"}}),
        {"cuda_ready": True},
    )
    check("an absurd learning rate warns",
          any(i["code"] == "high_lr" for i in bad_lr))


def test_auto_configure_gpu() -> None:
    section("Automatic configuration (simulated 24 GB NVIDIA tier)")
    fake_hardware = {
        "cuda_ready": True,
        "cuda": {
            "available": True,
            "bf16_supported": True,
            "devices": [{"name": "NVIDIA RTX 4090", "total_memory_mb": 24564}],
        },
        "gpu": {"available": True, "gpus": [{"memory_total_mb": 24564}]},
    }
    model_info = {
        "params": 1_240_000_000,
        "fields": {"num_hidden_layers": 16, "hidden_size": 2048, "max_position_embeddings": 8192},
    }
    dataset_report = {"stats": {"usable": 900, "max_chars": 4000}}
    result = config_mod.auto_configure(fake_hardware, model_info, dataset_report)
    cfg = result["config"]

    check("device is cuda", cfg["device"] == "cuda")
    check("precision is bf16 when supported", cfg["precision"] == "bf16")
    check("batch size is scaled to the gpu", cfg["batch_size"] >= 2, str(cfg["batch_size"]))
    check("context length adapts to the data",
          cfg["context_length"] <= 2048, str(cfg["context_length"]))
    check("lora alpha mirrors rank", cfg["lora_alpha"] == cfg["lora_r"] * 2)
    check("every choice has a reason", len(result["reasons"]) >= 5)
    check("reasons name the fields they explain",
          all("field" in reason and "reason" in reason for reason in result["reasons"]))

    fp16_hardware = json.loads(json.dumps(fake_hardware))
    fp16_hardware["cuda"]["bf16_supported"] = False
    fp16_cfg = config_mod.auto_configure(fp16_hardware, model_info, dataset_report)["config"]
    check("fp16 is chosen when bf16 is unsupported", fp16_cfg["precision"] == "fp16")


def test_auto_configure_cpu() -> None:
    section("Automatic configuration (no GPU)")
    result = config_mod.auto_configure({"cuda_ready": False, "cuda": {"devices": []}}, {}, {})
    cfg = result["config"]
    check("cpu device is selected", cfg["device"] == "cpu")
    check("fp32 on cpu", cfg["precision"] == "fp32")
    check("batch size of 1 on cpu", cfg["batch_size"] == 1)
    check("no quantization on cpu", cfg["quantization"] == "none")
    check("the cpu fallback is explained",
          any("CUDA" in reason["reason"] for reason in result["reasons"]))


def test_vram_estimator() -> None:
    section("VRAM estimator")
    model_info = {
        "params": 7_000_000_000,
        "params_exact": True,
        "fields": {"num_hidden_layers": 32, "hidden_size": 4096},
    }
    lora = estimator.estimate(
        config_mod.normalize({"method": "lora", "batch_size": 4, "context_length": 2048,
                              "precision": "bf16", "gradient_checkpointing": True}),
        model_info, available_vram_mb=24564,
    )
    check("estimate is available", lora["available"] is True)
    check("weights dominate for lora", lora["weights_mb"] > 10000, str(lora["weights_mb"]))
    check("lora trainable params are far below total",
          lora["trainable_params"] < lora["params"] / 100, str(lora["trainable_params"]))
    check("a verdict is given", lora["verdict"] in ("fits", "tight", "exceeds"))

    qlora = estimator.estimate(
        config_mod.normalize({"method": "qlora", "batch_size": 4, "context_length": 2048,
                              "precision": "bf16", "quantization": "4bit"}),
        model_info, available_vram_mb=24564,
    )
    check("4-bit weights are ~4x smaller than bf16",
          qlora["weights_mb"] < lora["weights_mb"] / 3, f"{qlora['weights_mb']} vs {lora['weights_mb']}")

    full = estimator.estimate(
        config_mod.normalize({"method": "full", "batch_size": 4, "context_length": 2048,
                              "precision": "bf16"}),
        model_info, available_vram_mb=8192,
    )
    check("full fine-tuning exceeds an 8 GB card", full["verdict"] == "exceeds", full["verdict"])
    check("exceeding offers concrete suggestions", len(full["suggestions"]) >= 2)
    check("the estimate states its assumptions", "note" in full["assumptions"])

    unknown = estimator.estimate(config_mod.normalize({}), {}, None)
    check("missing params yields an honest unknown", unknown["available"] is False)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def test_backend_helpers() -> None:
    """Pure helpers — no torch needed to verify them."""
    section("Backend compatibility helpers")
    from zeqouxtraining.backends.common import load_kwargs, supported_kwargs

    def signature_probe(a, b, c=1) -> None:  # noqa: ARG001
        pass

    filtered = supported_kwargs(signature_probe, {"a": 1, "b": 2, "nope": 3})
    check("unknown TrainingArguments kwargs are dropped",
          filtered == {"a": 1, "b": 2}, str(filtered))

    def loader_old(self, torch_dtype=None) -> None:  # noqa: ARG001
        pass

    def loader_new(self, dtype=None) -> None:  # noqa: ARG001
        pass

    kept = load_kwargs(loader_old, {"torch_dtype": "bf16"})
    check("older transformers keeps torch_dtype",
          kept == {"torch_dtype": "bf16"}, str(kept))

    renamed = load_kwargs(loader_new, {"torch_dtype": "bf16"})
    check("newer transformers gets dtype instead", renamed == {"dtype": "bf16"}, str(renamed))

    from zeqouxtraining.backends.registry import backend_capabilities

    caps = {c["name"]: c for c in backend_capabilities()}
    check("hf-peft declares the four methods",
          set(caps["hf-peft"]["methods"]) == {"lora", "qlora", "sft", "full"}, str(caps))
    check("hf-peft reports its missing packages instead of claiming readiness",
          isinstance(caps["hf-peft"]["missing"], list))


def test_checkpoints() -> None:
    section("Checkpoints")
    tmp = Path(tempfile.mkdtemp(prefix="zeqouxtraining-test-"))
    try:
        for step in (10, 20, 100):
            folder = tmp / f"checkpoint-{step}"
            folder.mkdir()
            (folder / "adapter_model.safetensors").write_bytes(b"0" * 64)
            (folder / "optimizer.pt").write_bytes(b"0" * 32)
            (folder / "trainer_state.json").write_text(json.dumps({
                "epoch": step / 10,
                "log_history": [{"step": step, "loss": 2.0 - step / 100}],
            }), encoding="utf-8")

        found = list_checkpoints(tmp)
        check("all checkpoints are found", len(found) == 3, str(len(found)))
        check("sorted by step, newest first", [c["step"] for c in found] == [100, 20, 10])
        check("loss is read from trainer_state", found[0]["loss"] == 1.0, str(found[0]["loss"]))
        check("optimizer presence is detected", found[0]["has_optimizer"] is True)
        check("size is measured from real files", found[0]["size_bytes"] > 90,
              str(found[0]["size_bytes"]))

        removed = prune(tmp, keep=1)
        check("prune keeps the newest", len(removed) == 2, str(removed))
        check("only one checkpoint remains", len(list_checkpoints(tmp)) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #

def test_error_humanizing() -> None:
    section("Error translation")
    cases = [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "cuda_oom"),
        (RuntimeError("Some weights are not used"), "unknown"),
        (ModuleNotFoundError("No module named 'peft'"), "missing_dependency"),
        (RuntimeError("401 Client Error: Unauthorized for url ..."), "gated_model"),
        (ConnectionError("Failed to establish a new connection"), "network"),
        (FileNotFoundError("model.safetensors"), "file_not_found"),
        (RuntimeError("size mismatch for model.embed_tokens.weight"), "state_dict_mismatch"),
    ]
    for exc, expected in cases:
        info = humanize(exc)
        check(f"{type(exc).__name__}({str(exc)[:34]!r}) -> {expected}",
              info["code"] == expected, f"got {info['code']}")
        check("  message is human readable", len(info["message"]) > 10 and "Traceback" not in info["message"])
        check("  a hint is provided", bool(info["hint"]))

    oom = humanize(RuntimeError("CUDA out of memory"))
    check("oom hint mentions concrete remedies",
          "batch size" in oom["hint"].lower(), oom["hint"])
    missing = humanize(ModuleNotFoundError("No module named 'torch'"))
    check("missing dependency hint points at Settings",
          "Settings" in missing["hint"], missing["hint"])


# --------------------------------------------------------------------------- #
# CLI protocol
# --------------------------------------------------------------------------- #

def _fake_artefact(root: Path, *, weights: bool = True, adapter: bool = True) -> Path:
    """Build a folder shaped like a finished run directory."""
    source = root / "demo-run"
    (source / "checkpoint-20").mkdir(parents=True, exist_ok=True)
    if adapter:
        (source / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        }), encoding="utf-8")
    else:
        (source / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{", encoding="utf-8")
    if weights:
        (source / "adapter_model.safetensors").write_bytes(b"0" * 4096)
    (source / "training.log").write_text("{}\n", encoding="utf-8")
    return source


def test_exporter() -> None:
    section("Export")
    from zeqouxtraining import exporter

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = _fake_artefact(root)

        info = exporter.describe(source)
        check("an adapter folder is recognised", info["is_adapter"] is True)
        check("the base model comes from adapter_config.json",
              info["base_model"] == "meta-llama/Llama-3.2-1B", info["base_model"])
        check("weights are found", len(info["weight_files"]) == 1, str(info["weight_files"]))
        check("checkpoints are listed", info["checkpoints"] == ["checkpoint-20"], str(info["checkpoints"]))
        check("the size is measured", info["size_bytes"] > 4000, str(info["size_bytes"]))
        check("copy mode is offered", info["modes"]["copy"]["ready"] is True)
        check("merge mode is recognised as applicable to an adapter",
              info["modes"]["merge"]["applicable"] is True)

        no_weights = _fake_artefact(root / "empty", weights=False)
        bare = exporter.describe(no_weights)
        check("a folder without weights reports a blocker",
              bare["modes"]["copy"]["ready"] is False and bare["blockers"], str(bare["blockers"]))

        try:
            exporter.describe(root / "nope")
            missing_ok = False
        except exporter.ExportError as exc:
            missing_ok = exc.code == "not_found" and bool(exc.hint)
        check("a missing export source explains itself", missing_ok)

        # ---- copy mode -------------------------------------------------
        target = root / "exported"
        summary = exporter.export(source, target, metadata={
            "name": "demo-run", "method": "lora", "baseModel": "meta-llama/Llama-3.2-1B",
            "finalLoss": 0.87, "steps": 240, "datasetName": "good.jsonl", "datasetRecords": 6,
        })
        check("copy export reports its mode", summary["mode"] == "copy" and summary["merged"] is False)
        check("adapter weights are copied", (target / "adapter_model.safetensors").exists())
        check("the adapter config is copied", (target / "adapter_config.json").exists())
        check("a manifest is written", (target / "export.json").exists())
        manifest = json.loads((target / "export.json").read_text(encoding="utf-8"))
        check("the manifest records the run",
              manifest["run"]["name"] == "demo-run" and manifest["mode"] == "copy")
        check("the manifest records the base model",
              manifest["base_model"] == "meta-llama/Llama-3.2-1B")
        check("checkpoint folders are not copied",
              not (target / "checkpoint-20").exists())

        readme = (target / "README.md").read_text(encoding="utf-8")
        check("a readable card is written", readme.startswith("# demo-run"), readme[:40])
        check("the card names the base model", "meta-llama/Llama-3.2-1B" in readme)
        check("the card shows the loss and steps", "0.87" in readme and "240" in readme)
        check("the card explains how to load the adapter", "PeftModel" in readme)
        check("the card is honest that the base model is not included",
              "The base model is not included" in readme)

        # ---- packed into one file --------------------------------------
        archive_target = root / "packed.zip"
        packed = exporter.export(source, archive_target, pack=True, metadata={"name": "demo-run"})
        check("a packed export reports itself as an archive",
              packed["archive"] is True and packed["mode"] == "copy", json.dumps(packed)[:110])
        check("a packed export is exactly one file",
              archive_target.exists() and archive_target.is_file())
        with zipfile.ZipFile(archive_target) as archive:
            names = sorted(archive.namelist())
            readme_txt = archive.read("README.md").decode("utf-8")
        check("the archive holds the weights and the config",
              "adapter_model.safetensors" in names and "adapter_config.json" in names, str(names))
        check("the archive carries the manifest and the card",
              "export.json" in names and "README.md" in names, str(names))
        check("checkpoint folders are not packed",
              not any(name.startswith("checkpoint-20") for name in names), str(names))
        check("the packed card explains how to unpack it", "unzip" in readme_txt, readme_txt[-160:])
        check("packing a folder name still produces one .zip",
              exporter.export(source, root / "named", pack=True)["output_dir"].endswith("named.zip"))

        # ---- refusals --------------------------------------------------
        try:
            exporter.export(source, target)
            overwrite_ok = False
        except exporter.ExportError as exc:
            overwrite_ok = exc.code == "not_empty"
        check("exporting into a used folder is refused", overwrite_ok)

        try:
            exporter.export(no_weights, root / "nowhere")
            empty_ok = False
        except exporter.ExportError as exc:
            empty_ok = exc.code == "no_weights"
        check("exporting a folder with no weights is refused", empty_ok)
        check("the refused target folder was not created", not (root / "nowhere").exists())

        full = _fake_artefact(root / "full", adapter=False)
        try:
            exporter.export(full, root / "merged", merge=True)
            merge_ok = False
        except exporter.ExportError as exc:
            merge_ok = exc.code == "not_an_adapter"
        check("merging a non-adapter is refused with a reason", merge_ok)

        if not exporter.runtime()["available"]:
            try:
                exporter.export(source, root / "merged", merge=True)
                deps_ok = False
            except exporter.ExportError as exc:
                deps_ok = exc.code == "missing_dependency" and "torch" in exc.message
            check("merging without the ML runtime names the missing packages", deps_ok)
            check("the merge refusal did not create the target", not (root / "merged").exists())

        # humanize must pass the curated message through untouched
        info_err = humanize(exporter.ExportError("Disk is full.", hint="Free some space.", code="disk_full"))
        check("export errors are not wrapped in a traceback message",
              info_err["message"] == "Disk is full." and info_err["code"] == "disk_full",
              json.dumps(info_err)[:120])
        check("export error hints survive", info_err["hint"] == "Free some space.")


def test_scratch_export_card() -> None:
    """Models that own every weight must not be described as adapters.

    A from-scratch run and a full fine-tune have no adapter to load on top of a
    base model, and a from-scratch run additionally carries the character-level
    tokenizer it learned — the exported card has to explain both, otherwise the
    folder looks broken to whoever opens it.
    """
    section("Export card for models without an adapter")
    from zeqouxtraining import exporter

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scratch = root / "scratch-run"
        scratch.mkdir()
        (scratch / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        (scratch / "model.safetensors").write_bytes(b"0" * 4096)
        (scratch / "zeqou_tokenizer.json").write_text(json.dumps({
            "tokenizer_class": "ZeqouCharTokenizer",
            "model_type": "char-level",
            "vocab": ["<pad>", "<unk>", "<bos>", "<eos>", "a", "b", " "],
        }), encoding="utf-8")

        info = exporter.describe(scratch)
        check("a from-scratch folder is not mistaken for an adapter", info["is_adapter"] is False)
        check("its weights are found",
              info["weight_files"] == ["model.safetensors"], str(info["weight_files"]))

        target = root / "exported-scratch"
        exporter.export(scratch, target, metadata={"name": "scratch-run", "method": "scratch"})
        readme = (target / "README.md").read_text(encoding="utf-8")
        check("the card calls it a standalone model", "standalone model" in readme, readme[-400:])
        check("the card never tells anyone to load an adapter", "PeftModel" not in readme)
        check("the card shows how to read the learned vocabulary",
              "vocab = json.load" in readme, readme[-400:])
        check("the exported folder keeps the character tokenizer",
              (target / "zeqou_tokenizer.json").exists())

        # A full fine-tune is standalone as well, but brings no learned vocab.
        full = root / "full-run"
        full.mkdir()
        (full / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        (full / "model.safetensors").write_bytes(b"0" * 4096)
        full_target = root / "exported-full"
        exporter.export(full, full_target, metadata={"name": "full-run", "method": "full"})
        full_readme = (full_target / "README.md").read_text(encoding="utf-8")
        check("a full fine-tune is described as standalone too",
              "standalone model" in full_readme, full_readme[-400:])
        check("no tokenizer section is invented for it",
              "vocab = json.load" not in full_readme, full_readme[-400:])


def test_cli_protocol() -> None:
    section("CLI protocol")

    def run(*args: str) -> list[dict]:
        proc = subprocess.run(
            [sys.executable, "-m", "zeqouxtraining.cli", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return [
            json.loads(line)
            for line in (proc.stdout or "").splitlines()
            if line.strip().startswith("{")
        ]

    events_out = run("validate-dataset", "--path", str(FIXTURES / "good.jsonl"),
                     "--context-length", "512")
    kinds = [event["event"] for event in events_out]
    check("validate-dataset ends with a result event", kinds[-1] == "result", str(kinds))
    check("stdout only carries protocol events",
          all(isinstance(event.get("detail"), dict) for event in events_out))

    fail_out = run("validate-dataset", "--path", str(FIXTURES / "missing-file.json"))
    check("a failed validation still returns a result payload",
          fail_out[-1]["event"] == "result" and fail_out[-1]["detail"]["ok"] is False,
          str(fail_out[-1]["detail"])[:120])

    no_source = run("validate-dataset")
    check("validate-dataset with no source reports a result, not a crash",
          no_source[-1]["event"] == "result"
          and no_source[-1]["detail"]["issues"][0]["code"] == "no_source",
          str(no_source[-1]["detail"])[:120])

    describe_out = run("export", "--source", str(FIXTURES), "--describe")
    check("export --describe reports what a folder holds",
          describe_out[-1]["event"] == "result" and "info" in describe_out[-1]["detail"],
          str(describe_out[-1]["detail"])[:120])

    export_files = run("export", "--source", str(FIXTURES), "--output", str(FIXTURES / "should-not-exist"))
    check("exporting a folder without weights fails with a readable message",
          export_files[-1]["event"] == "error"
          and "weight" in export_files[-1]["detail"]["message"].lower(),
          str(export_files[-1]["detail"])[:140])

    hf_out = run("validate-dataset", "--hf-id", "tatsu-lab/alpaca", "--split", "test")
    hf_detail = hf_out[-1]["detail"]
    check("--hf-id is accepted by the CLI",
          hf_detail["dataset"]["path"] == "tatsu-lab/alpaca",
          str(hf_detail["dataset"])[:120])

    backend_out = run("backends")
    names = [b["name"] for b in backend_out[-1]["detail"]["backends"]]
    check("the hf-peft backend is registered", "hf-peft" in names, str(names))
    check("the scratch backend is registered", "scratch" in names, str(names))
    scratch_cap = [b for b in backend_out[-1]["detail"]["backends"] if b["name"] == "scratch"][0]
    check("the scratch backend declares only the scratch method",
          scratch_cap["methods"] == ["scratch"], str(scratch_cap["methods"]))


def test_scratch_backend() -> None:
    """From-scratch training: architecture resolution, validation and routing."""
    section("Scratch backend (training from zero)")
    from zeqouxtraining.backends.scratch import CharTokenizer, SIZES, resolve_architecture

    arch = resolve_architecture({})
    check("the default architecture is the tiny preset",
          arch["layers"] == SIZES["tiny"]["layers"] and arch["hidden"] == SIZES["tiny"]["hidden"],
          str(arch))
    custom = resolve_architecture({"scratch_layers": 6, "scratch_hidden": 300, "scratch_heads": 8})
    check("explicit overrides win and heads are made to divide hidden evenly",
          custom["layers"] == 6 and custom["hidden"] == 300 and custom["hidden"] % custom["heads"] == 0,
          str(custom))

    backend = config_mod  # keep flake quiet about import order
    check("config module is importable", hasattr(backend, "METHODS"))

    # Validation: scratch needs no base model, but other methods still do.
    base_cfg = config_mod.normalize({"method": "scratch", "dataset": {"path": "x.jsonl"}})
    issues = config_mod.validate(base_cfg, {"cuda_ready": False})
    check("a scratch config validates without a base model",
          not any(i["code"] == "no_base_model" for i in issues),
          str([i["code"] for i in issues]))
    lora_cfg = config_mod.normalize({"method": "lora", "dataset": {"path": "x.jsonl"}})
    issues = config_mod.validate(lora_cfg, {"cuda_ready": False})
    check("lora still requires a base model",
          any(i["code"] == "no_base_model" for i in issues),
          str([i["code"] for i in issues]))

    char_tok = None
    try:
        char_tok = CharTokenizer(["hello world", "second line"], 5000)
    except Exception:
        char_tok = None
    check("the character tokenizer learns a vocabulary without any download",
          char_tok is not None and char_tok.vocab_size > 3,
          f"vocab {char_tok.vocab_size if char_tok else '?'}")
    if char_tok:
        encoded = char_tok(["hi"], truncation=True, max_length=8)
        check("the character tokenizer round-trips through decode",
              char_tok.decode(encoded["input_ids"][0]) == "hi")

    bad = subprocess.run(
        [sys.executable, "-m", "zeqouxtraining.cli", "not-a-command"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    check("an unknown command exits non-zero", bad.returncode != 0)


# --------------------------------------------------------------------------- #

def main() -> int:
    print("ZeqouXTraining backend tests")
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    test_events()
    test_deps_and_hardware()
    test_format_detection()
    test_good_dataset()
    test_translations()
    test_messy_dataset()
    test_broken_dataset()
    test_missing_file()
    test_chat_dataset()
    test_csv_and_txt()
    test_format_table()
    test_extra_formats()
    test_folder_dataset()
    test_huggingface_source()
    test_single_file_export()
    test_mapping_override()
    test_config_normalisation()
    test_config_validation()
    test_auto_configure_gpu()
    test_auto_configure_cpu()
    test_vram_estimator()
    test_backend_helpers()
    test_checkpoints()
    test_error_humanizing()
    test_exporter()
    test_scratch_export_card()
    test_cli_protocol()
    test_scratch_backend()

    print("\n" + "=" * 62)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    if FAILED:
        print("\nfailures:")
        for failure in FAILED:
            print(f"  - {failure}")
        return 1
    print("all backend tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
