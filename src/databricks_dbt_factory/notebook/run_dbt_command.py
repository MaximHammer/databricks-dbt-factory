# Databricks notebook source

import hashlib
import json
import os
import shlex
import shutil
import tempfile
import uuid
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from dbt.cli.main import dbtRunner

# COMMAND ----------

dbutils.widgets.text("dbt_commands", "")
dbutils.widgets.text("project_directory", "")
dbutils.widgets.text("profiles_directory", "")
dbutils.widgets.text("state_path", "")
dbutils.widgets.text("state_key", "")
dbutils.widgets.text("state_upstream", "")
dbutils.widgets.text("state_relation", "")

dbt_commands = dbutils.widgets.get("dbt_commands")
project_directory = dbutils.widgets.get("project_directory")
profiles_directory = dbutils.widgets.get("profiles_directory")
state_path = dbutils.widgets.get("state_path")
state_key = dbutils.widgets.get("state_key")
state_upstream = json.loads(dbutils.widgets.get("state_upstream") or "[]")
state_relation = json.loads(dbutils.widgets.get("state_relation") or "null")

if not dbt_commands:
    raise ValueError("dbt_commands parameter is required")

# COMMAND ----------

# Authenticate dbt with the notebook's own credentials via the Databricks SDK. The captured token
# is short-lived (about an hour) and does not self-refresh, so a single task running longer than its
# lifetime can fail mid-run.
ws = WorkspaceClient()
os.environ["DBT_ACCESS_TOKEN"] = ws.config.authenticate()["Authorization"].removeprefix("Bearer ").strip()

# dbt's host must be a bare hostname, but `config.host` is a full URL (scheme + host, and potentially
# a trailing slash or path). Parse it and keep only the netloc so a value like
# "https://my-workspace.databricks.com/" yields "my-workspace.databricks.com".
_parsed = urlparse(ws.config.host)
os.environ["DBT_HOST"] = _parsed.netloc or _parsed.path.strip("/")

# chdir to the dbt project so dbt runs from inside it. Relative `project_directory` is
# resolved against this notebook's own workspace location — the same anchor native
# `dbt_task` uses. Auto-copy mode sends `.` (resolves to the notebook's dir, which is
# project root by construction); user-pinned `--notebook-path` with relative
# `--project-directory` resolves against wherever the user placed the notebook; absolute
# `project_directory` is used as-is.
if project_directory:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_dir = os.path.dirname("/Workspace" + ctx.notebookPath().get())
    target_dir = (
        project_directory
        if os.path.isabs(project_directory)
        else os.path.normpath(os.path.join(notebook_dir, project_directory))
    )
    os.chdir(target_dir)

# Point dbt's artifact and log output at a private per-task dir so the parallel tasks of a job
# don't contend on the shared workspace `target/`/`logs/`.
local_dir = tempfile.mkdtemp(prefix="dbt_local_")
os.environ["DBT_TARGET_PATH"] = local_dir
os.environ["DBT_LOG_PATH"] = local_dir


# If a pre-built msgpack sits next to the project, deserialize it into a manifest and inject it into
# dbtRunner to skip dbt's parse phase (re-reading/hashing every file + DAG rebuild) on each task.
# Falls back to a normal parse if the msgpack is absent or unusable.
manifest = None
manifest_id = None
prebuilt_manifest_path = os.path.join("target", "partial_parse.msgpack")
if os.path.exists(prebuilt_manifest_path):
    try:
        from dbt.contracts.graph.manifest import Manifest
        from dbt.parser.manifest import extended_mashumuro_decoder

        with open(prebuilt_manifest_path, "rb") as f:
            prebuilt_manifest = f.read()
        manifest = Manifest.from_msgpack(prebuilt_manifest, decoder=extended_mashumuro_decoder)
        manifest.build_flat_graph()
        manifest_id = hashlib.sha256(prebuilt_manifest).hexdigest()
        print(f"[dbt-factory] injecting pre-built manifest from {prebuilt_manifest_path} (skipping dbt parse)")
    except Exception as e:
        print(f"[dbt-factory] manifest injection unavailable, falling back to dbt parse: {e}")
        manifest = None
        manifest_id = None

# COMMAND ----------

# With a state path, a task records after each successful build what its result depends on: the Unity Catalog
# `last_altered` of every table it reads, observed before the build so that a change landing mid-build counts as new;
# that of its own table, observed after the build so that a write by anything else shows; the deployed project; and its
# run context. It builds again only if one of those differs, a table it reads has no trustworthy value, or dbt's
# `state:modified+` selects it against the recorded project. Each value is compared only with its own earlier value,
# so no clocks are compared. A refresh moves the value of a materialized view even when nothing changed, while that of
# a streaming table fed by a continuous pipeline, an external table, a view or a federated table can miss writes, so
# such a source always counts as changed. Recording is best effort: a task whose build is not recorded builds again.
BUILD_COMMANDS = frozenset({"run", "seed", "snapshot", "test"})
TRUSTED_TABLE_TYPES = {"source": {"MANAGED", "MATERIALIZED_VIEW"}, "dbt": {"MANAGED", "EXTERNAL", "VIEW", "MATERIALIZED_VIEW"}}
CONTEXT_CHANGES = {
    "command": "its dbt command changed",
    "dbt": "the dbt version changed",
    "project": "dbt_project.yml changed",
    "env": "an env var the project reads changed",
}
task_record = os.path.join(state_path, "tasks", f"{state_key}.json") if state_path else None


def _target_args(args):
    for index, token in enumerate(args):
        if token in ("--target", "-t") and index + 1 < len(args):
            return [token, args[index + 1]]
        if token.startswith("--target=") or (token.startswith("-t") and not token.startswith("--") and len(token) > 2):
            return [token]
    return []


def _observe(relations, commands):
    names = ", ".join("(" + ", ".join(f"'{part}'" for part in relation[:3]) + ")" for relation in relations)
    query = (
        "select table_catalog, table_schema, table_name, table_type, unix_micros(last_altered) "
        f"from system.information_schema.tables where (table_catalog, table_schema, table_name) in ({names})"
    )
    build = next((args for args in commands if args[0] in BUILD_COMMANDS), [])
    profile = ["--profiles-dir", profiles_directory] if profiles_directory else []
    result = runner.invoke(
        ["show", "--inline", query, "--limit", "-1", "--log-level", "none", "--no-populate-cache", *_target_args(build), *profile]
    )
    if not result.success:
        print(f"[dbt-factory] the Unity Catalog lookup failed: {result.exception or result.result}")
        return None
    return {".".join(row[:3]): (row[3], None if row[4] is None else int(row[4])) for row in result.result.results[0].agate_table.rows}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _context(commands):
    versions = {}
    for distribution in ("dbt-core", "dbt-databricks"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    with open("dbt_project.yml", "rb") as f:
        project = hashlib.sha256(f.read()).hexdigest()
    return {
        "command": _digest(commands),
        "dbt": _digest(versions),
        "project": project,
        "env": _digest({name: os.environ.get(name) for name in manifest.env_vars}),
    }


def _read_record(path):
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and isinstance(record.get("relations"), dict) else None


def _build_reasons(observed, record, context):
    if record is None:
        return ["no successful build of this task is recorded"]
    reasons = [message for key, message in CONTEXT_CHANGES.items() if record.get(key) != context[key]]
    own = ".".join(state_relation)
    if own not in observed:
        reasons.append(f"{own} does not exist")
    elif observed[own][1] is None or record.get("own") != observed[own][1]:
        reasons.append(f"{own} changed since this task last built it")
    for *relation, kind in state_upstream:
        name = ".".join(relation)
        table_type, altered = observed.get(name, (None, None))
        if table_type is None:
            reasons.append(f"{name} was not found in Unity Catalog")
        elif table_type not in TRUSTED_TABLE_TYPES[kind] or altered is None:
            reasons.append(f"{name} is a {table_type} {kind}, whose last_altered can miss data changes")
        elif record["relations"].get(name) != altered:
            reasons.append(f"{name} changed")
    return reasons


def _select_modified(args, project):
    selected = list(args)
    for index, token in enumerate(args[:-1]):
        if token == "--select":
            selected[index + 1] = f"{args[index + 1]},state:modified+"
    return [*selected, "--state", project]


def _with_upstream_check(commands):
    if manifest_id is None:
        print("[dbt-factory] building: target/partial_parse.msgpack is missing or unreadable, so nothing identifies the deployed project")
        return commands, None
    observed = _observe([*state_upstream, state_relation], commands)
    if observed is None:
        print("[dbt-factory] building: the Unity Catalog lookup of the tables this task reads failed")
        return commands, None
    context = _context(commands)
    record = _read_record(task_record)
    pending = {**context, "relations": {name: observed[name][1] for name in (".".join(r[:3]) for r in state_upstream) if name in observed}}
    reasons = _build_reasons(observed, record, context)
    if reasons:
        print(f"[dbt-factory] building: {'; '.join(reasons)}")
        return commands, pending
    if record.get("manifest") == manifest_id:
        print("[dbt-factory] skipping: nothing this task depends on changed since its last successful build")
        return [], None
    project = os.path.join(state_path, "manifests", str(record.get("manifest")))
    if not record.get("manifest") or not os.path.exists(os.path.join(project, "manifest.json")):
        print("[dbt-factory] building: the project of this task's last build is not recorded")
        return commands, pending
    print(
        "[dbt-factory] nothing this task reads changed; building only what state:modified+ selects against the "
        "project of its last build"
    )
    return [_select_modified(args, project) if args[0] in BUILD_COMMANDS else args for args in commands], pending


def _write_atomically(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staged = f"{path}.{uuid.uuid4().hex}"
    with open(staged, "wb") as f:
        f.write(content)
    os.replace(staged, path)


def _prune_manifests():
    # A stored manifest stays while a task's record refers to it, so no task loses the project of its last build while
    # it runs the code check against it. A task whose recorded manifest is gone builds rather than run the code check.
    records = os.path.join(state_path, "tasks")
    referenced = {manifest_id} | {(_read_record(entry.path) or {}).get("manifest") for entry in os.scandir(records)}
    for entry in os.scandir(os.path.join(state_path, "manifests")):
        if entry.name not in referenced:
            shutil.rmtree(entry.path, ignore_errors=True)


def _record_build(commands, pending):
    own = ".".join(state_relation)
    try:
        after = _observe([state_relation], commands)
        if not after or own not in after:
            print(f"[dbt-factory] not recording this build: looking up {own} after it failed, so the next run builds again")
            return
        # dbt writes the project's manifest into the target path on every build command, even one that selects nothing.
        project = os.path.join(state_path, "manifests", manifest_id, "manifest.json")
        built = os.path.join(local_dir, "manifest.json")
        added = not os.path.exists(project) and os.path.exists(built)
        if added:
            with open(built, "rb") as f:
                _write_atomically(project, f.read())
        record = {"manifest": manifest_id if os.path.exists(project) else None, **pending, "own": after[own][1]}
        _write_atomically(task_record, json.dumps(record, indent=1).encode())
        print(f"[dbt-factory] recorded this build under {task_record}")
    except Exception as error:
        print(f"[dbt-factory] could not record this build, so the next run builds again: {error!r}")
        return
    if added:
        try:
            _prune_manifests()
        except OSError as error:
            print(f"[dbt-factory] could not remove the manifests no record refers to: {error!r}")


# COMMAND ----------


def _invoke(args):
    print(f"Running: dbt {' '.join(args)}")
    print("-" * 60)

    result = runner.invoke(args)

    if not result.success:
        detail = result.exception or result.result or "(no further details)"
        raise RuntimeError(f"dbt command failed: dbt {' '.join(args)}\n{detail}")

    print(f"Completed successfully: dbt {' '.join(args)}")
    return result


try:
    runner = dbtRunner(manifest=manifest)

    commands = []
    for command_str in json.loads(dbt_commands):
        command_str = command_str.strip()
        if not command_str:
            continue

        if command_str.startswith("dbt "):
            command_str = command_str[4:]

        args = shlex.split(command_str)

        if profiles_directory:
            args.extend(["--profiles-dir", profiles_directory])
        commands.append(args)

    pending = None
    if state_path:
        # The lookup parses an inline query, which needs the project's packages, so `dbt deps` runs first.
        while commands and commands[0][0] not in BUILD_COMMANDS:
            _invoke(commands.pop(0))
        commands, pending = _with_upstream_check(commands)

    for args in commands:
        result = _invoke(args)

    if pending is not None:
        _record_build(commands, pending)
finally:
    os.environ.pop("DBT_ACCESS_TOKEN", None)
    os.environ.pop("DBT_HOST", None)
    os.environ.pop("DBT_TARGET_PATH", None)
    os.environ.pop("DBT_LOG_PATH", None)
    # Remove the private per-task target/log dir; on reused (all-purpose) clusters these
    # would otherwise accumulate under the system temp dir for the life of the cluster.
    if local_dir:
        shutil.rmtree(local_dir, ignore_errors=True)
