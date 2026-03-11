#!/usr/bin/env python3
"""Conway-aware patch feature extractor for Experiment 5.1.

Extracts deep code-level signals from PR patches stored in prs_copy:
  - Import trust-boundary classification (tree-sitter, 10 languages)
  - Error-handling quality (regex on added lines)
  - Interface contract changes (function signature additions/changes)
  - Shared infrastructure stress (file path signals + content patterns)
  - Concurrency/resource management red flags
  - Schema & data contract changes
  - Observability coupling
  - Security trust boundary crossings

Outputs: data/conway_patch_features.jsonl  (one object per PR)
         data/conway_patch_features_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
OUT_JSONL    = os.path.join(ROOT, "data", "conway_patch_features.jsonl")
OUT_SUMMARY  = os.path.join(ROOT, "data", "conway_patch_features_summary.json")
REPOS_BASE   = "/shared_workspace_mfs/repos"

# ── Tree-sitter setup ──────────────────────────────────────────────────────

try:
    from tree_sitter import Language, Parser
    import tree_sitter_language_pack as tlp
    _TS_OK = True
except ImportError:
    _TS_OK = False
    print("WARNING: tree-sitter not available, import analysis will fall back to regex", file=sys.stderr)

# Languages we support and their file extensions
_LANG_BY_EXT: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts":   "typescript", ".tsx": "typescript",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".rb":   "ruby",   ".rake": "ruby",
    ".php":  "php",
    ".cpp":  "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hxx": "cpp",
    ".kt":   "kotlin", ".kts": "kotlin",
}

# Per-language stdlib top-level module names (sampled — not exhaustive)
_STDLIB: dict[str, set[str]] = {
    "python": {
        "os","sys","re","json","math","time","datetime","collections","itertools","functools",
        "pathlib","typing","abc","io","logging","threading","subprocess","hashlib","random",
        "string","copy","dataclasses","enum","contextlib","http","urllib","email","html",
        "xml","csv","pickle","struct","socket","asyncio","concurrent","multiprocessing",
        "inspect","importlib","warnings","traceback","unittest","argparse","configparser",
        "shutil","glob","fnmatch","tempfile","signal","errno","ctypes","platform","uuid",
        "base64","hmac","secrets","ssl","select","queue","heapq","bisect","array","weakref",
        "pprint","textwrap","unicodedata","codecs","locale","gettext","decimal","fractions",
        "statistics","cmath","numbers","operator","types","typing_extensions",
    },
    "javascript": {
        "path","fs","os","http","https","net","url","util","events","stream","buffer",
        "crypto","zlib","dns","child_process","cluster","readline","repl","vm","assert",
        "module","process","console","timers","string_decoder","querystring","domain",
        "tty","dgram","inspector","perf_hooks","v8","worker_threads","async_hooks",
    },
    "typescript": {  # same as JS
        "path","fs","os","http","https","net","url","util","events","stream","buffer",
        "crypto","zlib","child_process","readline","assert","module","process",
    },
    "go": {
        "fmt","os","io","net","log","sync","time","math","sort","strings","strconv",
        "bytes","bufio","errors","context","reflect","runtime","testing","flag","path",
        "regexp","unicode","encoding","crypto","hash","compress","archive","database",
        "html","text","image","mime","net/http","net/url","os/exec","io/ioutil",
        "io/fs","path/filepath","strings","strconv",
    },
    "rust": {
        "std","core","alloc",  # "std::" prefix is stdlib
    },
    "java": {
        "java","javax","sun","com.sun","jdk",  # prefixes
    },
    "ruby": {
        "set","json","csv","uri","net","date","time","fileutils","pathname","digest",
        "base64","openssl","tempfile","logger","forwardable","singleton","comparable",
        "enumerable","monitor","mutex_m","benchmark","pp","yaml","zlib","stringio",
        "shellwords","rbconfig","socket","timeout","open-uri","cgi","optparse",
    },
    "php": {
        # PHP builtins are functions, not imports; 'use' is namespaces
        # We flag anything that isn't a vendor/ path as potentially internal
    },
    "kotlin": {
        "kotlin","java","javax",
    },
    "cpp": {
        "iostream","fstream","sstream","string","vector","map","set","unordered_map",
        "unordered_set","algorithm","utility","memory","functional","thread","mutex",
        "condition_variable","atomic","chrono","cstdio","cstdlib","cstring","cmath",
        "cassert","climits","stdexcept","typeinfo","iterator","numeric","complex",
        "array","deque","list","queue","stack","tuple","optional","variant","any",
    },
}

_TS_PARSERS: dict[str, Parser] = {}

def _get_parser(lang: str) -> Optional[Parser]:
    if not _TS_OK:
        return None
    if lang not in _TS_PARSERS:
        try:
            language = tlp.get_language(lang)
            _TS_PARSERS[lang] = Parser(language)
        except Exception:
            _TS_PARSERS[lang] = None  # type: ignore
    return _TS_PARSERS.get(lang)

def _ts_nodes(node, *types):
    """Yield all descendant nodes matching any of the given types."""
    want = set(types)
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in want:
            yield cur
        if cur.children:
            stack.extend(reversed(cur.children))

# ── Import classification ──────────────────────────────────────────────────

@dataclass
class ImportCounts:
    total_new:    int = 0
    stdlib:       int = 0
    relative:     int = 0   # intra-repo
    external:     int = 0   # third-party (trust boundary crossing)
    # External sub-categories
    ext_network:  int = 0   # HTTP/DB/queue/rpc clients
    ext_security: int = 0   # crypto, auth, jwt, oauth
    ext_infra:    int = 0   # cloud SDKs, k8s, docker, terraform
    # Language
    lang:         str = ""


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[DiffHunk] = field(default_factory=list)

# External package name heuristics
_NETWORK_PKGS = re.compile(
    r"^(requests|httpx|aiohttp|axios|fetch|got|node-fetch|urllib3|grpc|"
    r"psycopg2?|sqlalchemy|pymysql|asyncpg|redis|pymongo|motor|cassandra|"
    r"elasticsearch|boto3|botocore|google-cloud|azure|kafka|pika|nats|"
    r"celery|kombu|amqplib|aiokafka|confluent.kafka|"
    r"django\.db|flask\.ext|fastapi|starlette|tornado\.httpclient)$",
    re.I,
)
_SECURITY_PKGS = re.compile(
    r"^(cryptography|paramiko|pyotp|jwt|pyjwt|bcrypt|passlib|"
    r"authlib|oauthlib|python-jose|itsdangerous|werkzeug\.security|"
    r"hashids|argon2|nacl)$",
    re.I,
)
_INFRA_PKGS = re.compile(
    r"^(boto3|botocore|google-cloud|azure|pulumi|terraform-|kubernetes|"
    r"docker|ansible|paramiko|fabric|invoke)$",
    re.I,
)

def _classify_pkg(top: str, lang: str) -> str:
    stdlib = _STDLIB.get(lang, set())
    # Language-specific stdlib checks
    if lang == "rust" and top in ("std", "core", "alloc"):
        return "stdlib"
    if lang == "java" and any(top.startswith(p) for p in ("java.", "javax.", "sun.", "jdk.")):
        return "stdlib"
    if lang == "kotlin" and any(top.startswith(p) for p in ("kotlin.", "java.", "javax.")):
        return "stdlib"
    if top in stdlib:
        return "stdlib"
    return "external"


def _parse_imports_python(added_src: bytes) -> ImportCounts:
    parser = _get_parser("python")
    counts = ImportCounts(lang="python")
    if parser is None:
        return counts
    try:
        tree = parser.parse(added_src)
    except Exception:
        return counts

    for node in _ts_nodes(tree.root_node, "import_statement"):
        for dn in _ts_nodes(node, "dotted_name"):
            top = dn.text.decode(errors="replace").split(".")[0]
            counts.total_new += 1
            kind = _classify_pkg(top, "python")
            if kind == "stdlib":
                counts.stdlib += 1
            else:
                counts.external += 1
                if _NETWORK_PKGS.match(top):   counts.ext_network  += 1
                if _SECURITY_PKGS.match(top):  counts.ext_security += 1
                if _INFRA_PKGS.match(top):     counts.ext_infra    += 1
            break  # only top-level name

    for node in _ts_nodes(tree.root_node, "import_from_statement"):
        text = node.text.decode(errors="replace")
        parts = text.split()
        module = parts[1] if len(parts) > 1 else ""
        counts.total_new += 1
        if module.startswith("."):
            counts.relative += 1
        else:
            top = module.split(".")[0]
            kind = _classify_pkg(top, "python")
            if kind == "stdlib":
                counts.stdlib += 1
            else:
                counts.external += 1
                if _NETWORK_PKGS.match(top):   counts.ext_network  += 1
                if _SECURITY_PKGS.match(top):  counts.ext_security += 1
                if _INFRA_PKGS.match(top):     counts.ext_infra    += 1

    return counts


def _parse_imports_js(added_src: bytes, lang: str = "javascript") -> ImportCounts:
    parser = _get_parser(lang)
    counts = ImportCounts(lang=lang)
    if parser is None:
        return counts
    try:
        tree = parser.parse(added_src)
    except Exception:
        return counts

    # ESM: import X from 'pkg' / import { X } from 'pkg'
    for node in _ts_nodes(tree.root_node, "import_statement"):
        for src_node in _ts_nodes(node, "string"):
            raw = src_node.text.decode(errors="replace").strip("'\"`")
            counts.total_new += 1
            if raw.startswith("."):
                counts.relative += 1
            else:
                top = raw.split("/")[0].lstrip("@")
                if _classify_pkg(top, lang) == "stdlib":
                    counts.stdlib += 1
                else:
                    counts.external += 1
                    if _NETWORK_PKGS.match(top):   counts.ext_network  += 1
                    if _SECURITY_PKGS.match(top):  counts.ext_security += 1
                    if _INFRA_PKGS.match(top):     counts.ext_infra    += 1
            break

    # CJS: require('pkg')
    for node in _ts_nodes(tree.root_node, "call_expression"):
        fn = node.child_by_field_name("function")
        if fn and fn.text == b"require":
            args = node.child_by_field_name("arguments")
            if args:
                for s in _ts_nodes(args, "string"):
                    raw = s.text.decode(errors="replace").strip("'\"`")
                    counts.total_new += 1
                    if raw.startswith("."):
                        counts.relative += 1
                    else:
                        top = raw.split("/")[0].lstrip("@")
                        if _classify_pkg(top, lang) == "stdlib":
                            counts.stdlib += 1
                        else:
                            counts.external += 1
                            if _NETWORK_PKGS.match(top):   counts.ext_network  += 1
                            if _SECURITY_PKGS.match(top):  counts.ext_security += 1
                            if _INFRA_PKGS.match(top):     counts.ext_infra    += 1
                    break
    return counts


def _parse_imports_go(added_src: bytes) -> ImportCounts:
    parser = _get_parser("go")
    counts = ImportCounts(lang="go")
    if parser is None:
        return counts
    try:
        tree = parser.parse(added_src)
    except Exception:
        return counts

    for node in _ts_nodes(tree.root_node, "import_spec"):
        for s in _ts_nodes(node, "interpreted_string_literal", "raw_string_literal"):
            raw = s.text.decode(errors="replace").strip('"`')
            counts.total_new += 1
            parts = raw.split("/")
            # stdlib: no dot in first segment (e.g. "fmt", "net/http")
            if "." not in parts[0]:
                counts.stdlib += 1
            else:
                counts.external += 1
                top = parts[0]
                if _NETWORK_PKGS.match(parts[-1]) or _NETWORK_PKGS.match(top):
                    counts.ext_network += 1
                if _INFRA_PKGS.match(top):
                    counts.ext_infra += 1
            break
    return counts


def _parse_imports_rust(added_src: bytes) -> ImportCounts:
    parser = _get_parser("rust")
    counts = ImportCounts(lang="rust")
    if parser is None:
        return counts
    try:
        tree = parser.parse(added_src)
    except Exception:
        return counts

    for node in _ts_nodes(tree.root_node, "use_declaration"):
        text = node.text.decode(errors="replace")
        # strip 'use ' prefix
        body = re.sub(r"^use\s+", "", text).rstrip(";").split("::")[0]
        counts.total_new += 1
        if body in ("crate", "super", "self"):
            counts.relative += 1
        elif body in ("std", "core", "alloc"):
            counts.stdlib += 1
        else:
            counts.external += 1
    return counts


def _parse_imports_java(added_src: bytes) -> ImportCounts:
    parser = _get_parser("java")
    counts = ImportCounts(lang="java")
    if parser is None:
        return counts
    try:
        tree = parser.parse(added_src)
    except Exception:
        return counts

    for node in _ts_nodes(tree.root_node, "import_declaration"):
        text = node.text.decode(errors="replace")
        m = re.match(r"import\s+(?:static\s+)?([\w\.]+)", text)
        if not m:
            continue
        fqn = m.group(1)
        top2 = ".".join(fqn.split(".")[:2])
        counts.total_new += 1
        if any(fqn.startswith(p) for p in ("java.", "javax.", "sun.", "jdk.", "com.sun.")):
            counts.stdlib += 1
        else:
            counts.external += 1
            if _NETWORK_PKGS.match(fqn.split(".")[-1]):
                counts.ext_network += 1
    return counts


def _parse_imports_generic_regex(added_lines: list[str], lang: str) -> ImportCounts:
    """Fallback: regex-based import parsing for Ruby, PHP, Kotlin, C++."""
    counts = ImportCounts(lang=lang)
    stdlib = _STDLIB.get(lang, set())

    patterns = {
        "ruby":   re.compile(r"^require(?:_relative)?\s+['\"]([^'\"]+)['\"]"),
        "php":    re.compile(r"^use\s+([\w\\]+)"),
        "kotlin": re.compile(r"^import\s+([\w\.]+)"),
        "cpp":    re.compile(r'^#include\s+[<"]([^>"]+)[>"]'),
    }
    pat = patterns.get(lang)
    if not pat:
        return counts

    for line in added_lines:
        m = pat.match(line.strip())
        if not m:
            continue
        pkg = m.group(1)
        counts.total_new += 1
        if lang == "ruby":
            if pkg.startswith("."):
                counts.relative += 1
            elif pkg in stdlib:
                counts.stdlib += 1
            else:
                counts.external += 1
        elif lang == "cpp":
            top = pkg.split("/")[0]
            if top in stdlib:
                counts.stdlib += 1
            else:
                counts.external += 1
        elif lang == "kotlin":
            top2 = ".".join(pkg.split(".")[:2])
            if any(pkg.startswith(p) for p in ("kotlin.", "java.", "javax.")):
                counts.stdlib += 1
            else:
                counts.external += 1
        else:
            counts.external += 1  # PHP: assume external unless can tell
    return counts


# ── Regex-based signal patterns ────────────────────────────────────────────

# Error handling
_TRY_CATCH      = re.compile(r"^\+\s*(try\s*[\{:]|catch\s*[\(\{]|except[\s\(:@]|except$)", re.M)
_BARE_EXCEPT    = re.compile(r"^\+\s*(except\s*:|catch\s*\(\s*\)\s*\{?|catch\s*\{)", re.M)
_EXCEPT_PASS    = re.compile(r"^\+\s*except[^\n]*:\s*\n\s*pass", re.M)
_FINALLY        = re.compile(r"^\+\s*(finally\s*[\{:])", re.M)
_RAISE_FROM     = re.compile(r"^\+.+raise .+ from ", re.M)
_LOG_IN_EXCEPT  = re.compile(r"^\+.*(logger\.|logging\.|log\.|console\.error|System\.err)", re.M)
_RERAISE        = re.compile(r"^\+\s*(raise$|throw;|throw err|throw error|throw e\b)", re.M)

# External client introductions
_HTTP_CLIENT    = re.compile(r"^\+.*(requests\.(get|post|put|delete|patch|Session)\b|axios\.|"
                              r"fetch\s*\(|httpx\.(get|post|AsyncClient)|aiohttp\.|urllib\.request|"
                              r"http\.client\.|okhttp\.|RestTemplate\b|WebClient\.)", re.M)
_HTTP_NO_TIMEOUT= re.compile(r"requests\.(get|post|put|delete|patch)\s*\([^)]*\)", re.M)
_HTTP_TIMEOUT   = re.compile(r"timeout\s*=", re.M)
_DB_CLIENT      = re.compile(r"^\+.*(\.connect\s*\(|createConnection\s*\(|sessionmaker\s*\(|"
                              r"redis\.(Redis|StrictRedis|from_url)\s*\(|"
                              r"psycopg2?\.connect|pymysql\.connect|asyncpg\.connect|"
                              r"MongoClient\s*\(|motor\.|cassandra\.cluster|"
                              r"elasticsearch\.Elasticsearch\s*\(|"
                              r"create_engine\s*\(|DriverManager\.getConnection)", re.M)
_DB_POOL_CONFIG = re.compile(r"(pool_size|max_overflow|pool_timeout|maxPoolSize|minConnections|"
                              r"connectionLimit|max_connections)", re.M)
_QUEUE_CLIENT   = re.compile(r"^\+.*(KafkaProducer|KafkaConsumer|pika\.BlockingConnection|"
                              r"boto3.*sqs|SQSClient|RabbitMQ|nats\.connect|"
                              r"aiokafka\.|confluent_kafka\.)", re.M)

# Resource management
_CONTEXT_MGR    = re.compile(r"^\+\s*with\s+", re.M)
_OPEN_NO_WITH   = re.compile(r"^\+(?!.*with\b).*\bopen\s*\(", re.M)
_THREAD_SPAWN   = re.compile(r"^\+.*(threading\.Thread\s*\(|asyncio\.create_task\s*\(|"
                              r"go\s+func\s*\(|new\s+Thread\s*\(|\.start\(\)\s*$|"
                              r"executor\.submit\()", re.M)
_LOCK_PRIM      = re.compile(r"^\+.*(threading\.(Lock|RLock|Semaphore)\(\)|"
                              r"asyncio\.(Lock|Semaphore)\(\)|sync\.Mutex|sync\.RWMutex|"
                              r"synchronized\s*\()", re.M)
_GLOBAL_KW      = re.compile(r"^\+\s*global\s+\w+", re.M)

# Security
_SUBPROCESS_SHELL = re.compile(r"^\+.*subprocess\.(run|call|Popen|check_output).*shell\s*=\s*True", re.M)
_EVAL_EXEC       = re.compile(r"^\+.*\b(eval\s*\(|exec\s*\(|__import__\s*\(|compile\s*\()", re.M)
_PICKLE_LOADS    = re.compile(r"^\+.*(pickle\.loads|yaml\.load\s*\([^,)]+\)(?!\s*,\s*Loader))", re.M)
_HARDCODED_CRED  = re.compile(r'^\+.*(password|secret|api_key|token|passwd)\s*=\s*["\'][^"\']{4,}["\']', re.M | re.I)
_CRED_IN_LOG     = re.compile(r'^\+.*(log|print)\b.*\b(password|token|secret|api_key)\b', re.M | re.I)
_SQL_FSTRING     = re.compile(r'^\+.*(?:f["\']|%\s*["\']|\.format\s*\().*(?:SELECT|INSERT|UPDATE|DELETE|WHERE)', re.M | re.I)
_INPUT_VALIDATE  = re.compile(r"^\+.*(pydantic\.|cerberus\.|marshmallow\.|jsonschema\.|"
                               r"validate\s*\(|@validator|@field_validator)", re.M)

# Observability
_METRIC_EMIT     = re.compile(r"^\+.*(prometheus|statsd|datadog|opentelemetry|"
                               r"Counter\s*\(|Histogram\s*\(|Gauge\s*\(|"
                               r"\.increment\s*\(|\.timing\s*\(|\.gauge\s*\()", re.M)
_LOG_WARN_ERR    = re.compile(r"^\+.*(logging\.(warning|error|critical|exception)\s*\(|"
                               r"logger\.(warn|error|critical|exception)\s*\(|"
                               r"log\.(warn|error|critical)\s*\(|"
                               r"console\.(error|warn)\s*\()", re.M)
_HEALTH_CHECK    = re.compile(r"^\+.*(health_check|healthz|readyz|liveness|readiness|"
                               r"ping\s*\(\s*\)|is_alive|is_connected)", re.M)

# Interface contracts
_NEW_FUNC_DEF    = re.compile(r"^\+\s*(async\s+)?def\s+(\w+)\s*\(", re.M)
_PUB_FUNC_DEF    = re.compile(r"^\+\s*(public|export\s+(default\s+)?function|"
                               r"func\s+[A-Z]\w*\s*\(|pub\s+fn\s+\w+)", re.M)
_DEPRECATED      = re.compile(r"^\+.*(@deprecated|@Deprecated|#\[deprecated|DeprecationWarning)", re.M)
_VERSION_SUFFIX  = re.compile(r"^\+.*(def |class |func |fn )\w*(v\d+|V\d+|_v\d+)\s*[\(\{]", re.M)

# Shared infrastructure
_SHARED_UTIL     = re.compile(r"(^|/)(utils?|common|shared|lib|core|helpers?|mixins?|base)/", re.I)
_AUTH_CODE       = re.compile(r"(authenticate|authorize|permission|jwt|oauth|token_required|"
                               r"@login_required|@requires_auth)", re.I)
_STARTUP_FILE    = re.compile(r"(__init__\.py|main\.py|app\.py|server\.py|manage\.py|"
                               r"wsgi\.py|asgi\.py|settings\.py|config\.py)$", re.I)
_LOG_FATAL       = re.compile(r"^\+.*(os\.exit\s*\(|log\.fatal\s*\(|log\.Fatal\s*\(|"
                               r"sys\.exit\s*\(0\)|System\.exit\s*\()", re.M)

# Schema / data contract
_SCHEMA_CHANGE   = re.compile(r"^\+.*(ALTER\s+TABLE|CREATE\s+TABLE|DROP\s+TABLE|ADD\s+COLUMN|"
                               r"models\.\w+Field\s*\(|Column\s*\(|db\.Column\s*\()", re.M | re.I)
_NON_NULLABLE_COL= re.compile(r"^\+.*(NOT\s+NULL|nullable\s*=\s*False|null=False)", re.M | re.I)

# Dependency / manifest files
_DEP_FILE        = re.compile(r"(requirements.*\.txt|Pipfile$|Pipfile\.lock|"
                               r"package\.json|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|"
                               r"Cargo\.toml|Cargo\.lock|go\.mod|go\.sum|"
                               r"pom\.xml|build\.gradle|build\.gradle\.kts|"
                               r"[Gg]emfile|[Gg]emfile\.lock|setup\.cfg|pyproject\.toml|"
                               r"poetry\.lock|setup\.py|composer\.json|composer\.lock)$")
_MIGRATION_FILE  = re.compile(r"(migrations?/|alembic/|flyway/|liquibase/|V\d+__.*\.sql)", re.I)
_CONFIG_FILE     = re.compile(r"\.(ya?ml|toml|ini|conf|cfg|env|properties)$", re.I)
_INFRA_FILE      = re.compile(r"(dockerfile|docker-compose|\.tf$|\.tfvars|"
                               r"\.k8s\.|kubernetes|helm|chart\.yaml|values\.yaml|"
                               r"\.github/workflows/|\.gitlab-ci\.|jenkins|buildkite)", re.I)

# ── Per-patch extraction ───────────────────────────────────────────────────

# Safety limits to prevent catastrophic regex backtracking on huge/minified patches
_MAX_PATCH_BYTES = 300_000   # truncate patch before regex scanning
_MAX_LINE_LEN    = 500       # truncate each added line (prevents O(n²) with double .*)

def _added_lines(patch: str) -> list[str]:
    lines = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    # Truncate long lines to prevent catastrophic backtracking in patterns like .*X.*Y
    return [l[:_MAX_LINE_LEN] for l in lines]


def _filenames(fp_raw) -> list[str]:
    try:
        fp = json.loads(fp_raw) if isinstance(fp_raw, str) else fp_raw
        if isinstance(fp, list):
            names = []
            for item in fp:
                if isinstance(item, dict):
                    names.append(item.get("file") or item.get("filename") or item.get("path") or "")
                elif isinstance(item, str):
                    names.append(item)
            return [n for n in names if n]
    except Exception:
        pass
    return []


def _detect_langs(fnames: list[str]) -> Counter:
    lang_count: Counter = Counter()
    for f in fnames:
        ext = os.path.splitext(f)[1].lower()
        lang = _LANG_BY_EXT.get(ext)
        if lang:
            lang_count[lang] += 1
    return lang_count


def _parse_imports_for_lang(added_src: bytes, lang: str, added_lines: list[str]) -> ImportCounts:
    if lang == "python":
        return _parse_imports_python(added_src)
    elif lang in ("javascript", "typescript"):
        return _parse_imports_js(added_src, lang)
    elif lang == "go":
        return _parse_imports_go(added_src)
    elif lang == "rust":
        return _parse_imports_rust(added_src)
    elif lang == "java":
        return _parse_imports_java(added_src)
    elif lang in ("ruby", "php", "kotlin", "cpp"):
        return _parse_imports_generic_regex(added_lines, lang)
    return ImportCounts(lang=lang)


def _repo_dir_map() -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.isdir(REPOS_BASE):
        return out
    for entry in os.listdir(REPOS_BASE):
        path = os.path.join(REPOS_BASE, entry)
        if not os.path.isdir(path):
            continue
        parts = entry.split("__")
        if len(parts) < 3:
            continue
        owner = parts[1]
        name = "__".join(parts[2:])
        out[f"{owner}/{name}"] = path
    return out


def _normalize_diff_path(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_unified_diff(patch: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: DiffHunk | None = None
    for raw in patch.splitlines():
        if raw.startswith("diff --git "):
            parts = raw.split()
            old_path = _normalize_diff_path(parts[2]) if len(parts) > 2 else ""
            new_path = _normalize_diff_path(parts[3]) if len(parts) > 3 else old_path
            current = FileDiff(old_path=old_path, new_path=new_path)
            files.append(current)
            current_hunk = None
            continue
        if current is None:
            continue
        if raw.startswith("--- "):
            current.old_path = "" if raw[4:] == "/dev/null" else _normalize_diff_path(raw[4:])
            continue
        if raw.startswith("+++ "):
            current.new_path = "" if raw[4:] == "/dev/null" else _normalize_diff_path(raw[4:])
            continue
        m = _HUNK_RE.match(raw)
        if m:
            current_hunk = DiffHunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2) or "1"),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or "1"),
            )
            current.hunks.append(current_hunk)
            continue
        if current_hunk is not None and raw[:1] in {" ", "+", "-", "\\"}:
            current_hunk.lines.append(raw)
    return files


def _git_show_text(repo_dir: str, sha: str, path: str) -> str:
    if not repo_dir or not sha or not path:
        return ""
    rr = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo_dir, "show", f"{sha}:{path}"],
        capture_output=True,
        text=True,
    )
    return rr.stdout if rr.returncode == 0 else ""


def _read_workspace_text(workspace_dir: str | None, path: str) -> str:
    if not workspace_dir or not path:
        return ""
    abs_path = os.path.join(workspace_dir, path)
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        return ""
    try:
        return open(abs_path, errors="replace").read()
    except Exception:
        return ""


def _apply_file_hunks(before_text: str, file_diff: FileDiff) -> str:
    if not file_diff.new_path:
        return ""
    before_lines = before_text.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0
    for hunk in file_diff.hunks:
        start_old = max(0, hunk.old_start - 1)
        out.extend(before_lines[cursor:start_old])
        idx = start_old
        no_newline_next = False
        for i, raw in enumerate(hunk.lines):
            prefix = raw[:1]
            body = raw[1:]
            if prefix == "\\":
                no_newline_next = True
                continue
            if prefix == " ":
                if idx < len(before_lines):
                    out.append(before_lines[idx])
                else:
                    out.append(body + ("" if no_newline_next else "\n"))
                idx += 1
            elif prefix == "-":
                idx += 1
            elif prefix == "+":
                out.append(body + ("" if no_newline_next else "\n"))
            no_newline_next = False
        cursor = idx
    out.extend(before_lines[cursor:])
    return "".join(out)


_AST_METRIC_KEYS = [
    "func_defs",
    "class_defs",
    "public_defs",
    "param_total",
    "call_sites",
    "branch_nodes",
    "try_nodes",
    "import_nodes",
    "inheritance_nodes",
    "assignment_nodes",
]

_FUNC_NODE_TYPES = {
    "function_definition", "function_declaration", "method_definition", "method_declaration",
    "function_item", "method", "function", "func_literal", "constructor_declaration",
}
_CLASS_NODE_TYPES = {
    "class_definition", "class_declaration", "class_specifier", "struct_item",
    "enum_item", "interface_declaration", "object_declaration", "trait_item",
}
_CALL_NODE_TYPES = {"call", "call_expression", "method_invocation"}
_BRANCH_NODE_TYPES = {
    "if_statement", "if_expression", "for_statement", "for_expression", "while_statement",
    "while_expression", "switch_statement", "switch_expression", "match_expression",
    "case_statement", "case_clause", "select_statement",
}
_TRY_NODE_TYPES = {"try_statement", "except_clause", "catch_clause", "finally_clause"}
_IMPORT_NODE_TYPES = {
    "import_statement", "import_from_statement", "import_declaration",
    "use_declaration", "preproc_include", "include_statement",
}
_INHERIT_NODE_TYPES = {"extends_clause", "implements_clause", "super_interfaces", "superclass", "base_class_clause"}
_ASSIGN_NODE_TYPES = {"assignment", "assignment_expression"}
_PARAM_NODE_TYPES = {
    "parameters", "formal_parameters", "parameter_list", "lambda_parameters",
    "typed_parameter", "required_parameter", "optional_parameter", "parameter",
}


def _zero_ast_metrics() -> dict[str, int]:
    return {k: 0 for k in _AST_METRIC_KEYS}


def _is_public_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _ast_metrics_for_source(source: str, path: str) -> dict[str, int]:
    ext = os.path.splitext(path)[1].lower()
    lang = _LANG_BY_EXT.get(ext)
    if not source or not lang:
        return _zero_ast_metrics()
    parser = _get_parser(lang)
    if parser is None:
        return _zero_ast_metrics()
    try:
        tree = parser.parse(source.encode(errors="replace"))
    except Exception:
        return _zero_ast_metrics()

    metrics = _zero_ast_metrics()
    for node in _ts_nodes(tree.root_node, *_FUNC_NODE_TYPES):
        metrics["func_defs"] += 1
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode(errors="replace") if name_node is not None else ""
        if _is_public_name(name):
            metrics["public_defs"] += 1
    for node in _ts_nodes(tree.root_node, *_CLASS_NODE_TYPES):
        metrics["class_defs"] += 1
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode(errors="replace") if name_node is not None else ""
        if _is_public_name(name):
            metrics["public_defs"] += 1
    for node in _ts_nodes(tree.root_node, *_PARAM_NODE_TYPES):
        metrics["param_total"] += max(1, len([c for c in node.named_children if c.type not in {"block", "type_annotation"}]))
    metrics["call_sites"] = sum(1 for _ in _ts_nodes(tree.root_node, *_CALL_NODE_TYPES))
    metrics["branch_nodes"] = sum(1 for _ in _ts_nodes(tree.root_node, *_BRANCH_NODE_TYPES))
    metrics["try_nodes"] = sum(1 for _ in _ts_nodes(tree.root_node, *_TRY_NODE_TYPES))
    metrics["import_nodes"] = sum(1 for _ in _ts_nodes(tree.root_node, *_IMPORT_NODE_TYPES))
    metrics["inheritance_nodes"] = sum(1 for _ in _ts_nodes(tree.root_node, *_INHERIT_NODE_TYPES))
    metrics["assignment_nodes"] = sum(1 for _ in _ts_nodes(tree.root_node, *_ASSIGN_NODE_TYPES))
    return metrics


def _path_metrics(fnames: list[str]) -> dict[str, float]:
    if not fnames:
        return {
            "path_depth_mean": 0.0,
            "path_depth_max": 0.0,
            "topdir_entropy": 0.0,
            "test_file_ratio": 0.0,
            "docs_file_ratio": 0.0,
        }
    depths = [max(0, f.count("/")) for f in fnames]
    topdirs = Counter((f.split("/")[0] if "/" in f else f) for f in fnames)
    total = float(len(fnames))
    entropy = 0.0
    for c in topdirs.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)
    test_files = sum(1 for f in fnames if re.search(r"(^|/)(tests?|spec|specs|__tests__)(/|$)|(_test|\\.spec|\\.test)\\.", f, re.I))
    docs_files = sum(1 for f in fnames if re.search(r"(^|/)(docs?|documentation)(/|$)|README|CHANGELOG|NEWS", f, re.I))
    return {
        "path_depth_mean": float(sum(depths) / len(depths)),
        "path_depth_max": float(max(depths)),
        "topdir_entropy": float(entropy),
        "test_file_ratio": float(test_files / total),
        "docs_file_ratio": float(docs_files / total),
    }


def _blame_metrics(repo_dir: str | None, base_sha: str | None, file_diff: FileDiff) -> dict[str, float]:
    if not repo_dir or not base_sha or not file_diff.old_path or not file_diff.hunks:
        return {
            "blame_unique_authors": 0.0,
            "blame_top_author_share": 0.0,
            "blame_author_entropy": 0.0,
            "blame_multi_author_hunks": 0.0,
        }
    rr = subprocess.run(
        [
            "git", "-c", "safe.directory=*", "-C", repo_dir, "blame", "--line-porcelain",
            base_sha, "--", file_diff.old_path,
        ],
        capture_output=True,
        text=True,
    )
    if rr.returncode != 0:
        return {
            "blame_unique_authors": 0.0,
            "blame_top_author_share": 0.0,
            "blame_author_entropy": 0.0,
            "blame_multi_author_hunks": 0.0,
        }

    line_to_author: dict[int, str] = {}
    current_author = ""
    current_line = None
    for line in rr.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{7,40}", parts[0]):
            try:
                current_line = int(parts[2])
            except Exception:
                current_line = None
            continue
        if line.startswith("author-mail "):
            current_author = line.split(" ", 1)[1].strip()
            if current_line is not None:
                line_to_author[current_line] = current_author
                current_line = current_line + 1

    counts: Counter = Counter()
    multi_author_hunks = 0
    for hunk in file_diff.hunks:
        if hunk.old_count <= 0:
            continue
        start = hunk.old_start
        end = hunk.old_start + hunk.old_count - 1
        hunk_authors: Counter = Counter()
        for ln in range(start, end + 1):
            author = line_to_author.get(ln)
            if not author:
                continue
            counts[author] += 1
            hunk_authors[author] += 1
        if len(hunk_authors) > 1:
            multi_author_hunks += 1
    total = sum(counts.values())
    if total <= 0:
        return {
            "blame_unique_authors": 0.0,
            "blame_top_author_share": 0.0,
            "blame_author_entropy": 0.0,
            "blame_multi_author_hunks": 0.0,
        }
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)
    return {
        "blame_unique_authors": float(len(counts)),
        "blame_top_author_share": float(max(counts.values()) / total),
        "blame_author_entropy": float(entropy),
        "blame_multi_author_hunks": float(multi_author_hunks),
    }


def _contextual_patch_features(
    patch: str,
    fnames: list[str],
    repo_dir: str | None = None,
    base_sha: str | None = None,
    workspace_dir: str | None = None,
) -> dict[str, float]:
    patch_files = _parse_unified_diff(patch)
    agg = Counter()
    blame_acc = Counter()
    public_api_file_count = 0
    supported_files = 0
    blamed_files = 0
    for file_diff in patch_files:
        path = file_diff.new_path or file_diff.old_path
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in _LANG_BY_EXT:
            continue
        supported_files += 1
        before_text = _git_show_text(repo_dir or "", base_sha or "", file_diff.old_path or path)
        after_text = _read_workspace_text(workspace_dir, file_diff.new_path or path)
        if not after_text:
            after_text = _apply_file_hunks(before_text, file_diff)
        before_ast = _ast_metrics_for_source(before_text, path)
        after_ast = _ast_metrics_for_source(after_text, path)
        if after_ast["public_defs"] > 0:
            public_api_file_count += 1
        for key in _AST_METRIC_KEYS:
            agg[f"ast_before_{key}"] += before_ast[key]
            agg[f"ast_after_{key}"] += after_ast[key]
            agg[f"ast_delta_{key}"] += after_ast[key] - before_ast[key]
        needs_blame = (
            bool(_SHARED_UTIL.search(path))
            or bool(_AUTH_CODE.search(path))
            or bool(_STARTUP_FILE.search(path))
            or after_ast["public_defs"] > 0
            or before_ast["public_defs"] > 0
            or (after_ast["public_defs"] - before_ast["public_defs"]) != 0
        )
        before_line_count = before_text.count("\n") + 1 if before_text else 0
        touched_old_lines = sum(max(0, h.old_count) for h in file_diff.hunks)
        if needs_blame and before_line_count <= 3000 and touched_old_lines <= 1200:
            blame_stats = _blame_metrics(repo_dir, base_sha, file_diff)
            if any(v > 0 for v in blame_stats.values()):
                blamed_files += 1
            for key, value in blame_stats.items():
                blame_acc[key] += value

    path_stats = _path_metrics(fnames)
    n_files = max(1, len(fnames))
    ast_delta_public_defs = int(agg.get("ast_delta_public_defs", 0))
    out: dict[str, float] = {
        **{k: float(v) for k, v in agg.items()},
        "ast_supported_files": float(supported_files),
        "public_api_file_ratio": float(public_api_file_count / n_files),
        **path_stats,
    }
    if blamed_files > 0:
        out.update(
            {
                "blame_unique_authors": float(blame_acc["blame_unique_authors"] / blamed_files),
                "blame_top_author_share": float(blame_acc["blame_top_author_share"] / blamed_files),
                "blame_author_entropy": float(blame_acc["blame_author_entropy"] / blamed_files),
                "blame_multi_author_hunks": float(blame_acc["blame_multi_author_hunks"]),
                "blamed_files": float(blamed_files),
            }
        )
    else:
        out.update(
            {
                "blame_unique_authors": 0.0,
                "blame_top_author_share": 0.0,
                "blame_author_entropy": 0.0,
                "blame_multi_author_hunks": 0.0,
                "blamed_files": 0.0,
            }
        )
    return out


def extract_features(
    patch: str,
    fp_raw,
    pr_merged: bool,
    total_review_threads: int,
    total_comments: int,
    review_threads_json: str,
    repo_dir: str | None = None,
    base_sha: str | None = None,
    workspace_dir: str | None = None,
) -> dict:
    # Truncate patch to avoid catastrophic backtracking on huge diffs
    if len(patch) > _MAX_PATCH_BYTES:
        patch = patch[:_MAX_PATCH_BYTES]
    fnames = _filenames(fp_raw)
    added = _added_lines(patch)
    added_src = "\n".join(added).encode(errors="replace")

    # ── Language detection ────────────────────────────────────────────────
    lang_counts = _detect_langs(fnames)
    primary_lang = lang_counts.most_common(1)[0][0] if lang_counts else "unknown"
    context_feats = _contextual_patch_features(
        patch,
        fnames,
        repo_dir=repo_dir,
        base_sha=base_sha,
        workspace_dir=workspace_dir,
    )

    # ── Import analysis (tree-sitter) ─────────────────────────────────────
    imp = _parse_imports_for_lang(added_src, primary_lang, added)

    # Aggregate across all languages in multi-lang PRs
    if len(lang_counts) > 1:
        for lang in list(lang_counts.keys())[1:4]:   # up to 4 extra langs
            extra = _parse_imports_for_lang(added_src, lang, added)
            imp.total_new    += extra.total_new
            imp.stdlib       += extra.stdlib
            imp.relative     += extra.relative
            imp.external     += extra.external
            imp.ext_network  += extra.ext_network
            imp.ext_security += extra.ext_security
            imp.ext_infra    += extra.ext_infra

    # ── File-based signals ────────────────────────────────────────────────
    has_dep_file     = any(_DEP_FILE.search(f)       for f in fnames)
    has_migration    = any(_MIGRATION_FILE.search(f) for f in fnames)
    has_config_file  = any(_CONFIG_FILE.search(f)    for f in fnames)
    has_infra_file   = any(_INFRA_FILE.search(f)     for f in fnames)
    has_startup_file = any(_STARTUP_FILE.search(f)   for f in fnames)
    modifies_shared  = any(_SHARED_UTIL.search(f)    for f in fnames)
    modifies_auth    = any(_AUTH_CODE.search(f)      for f in fnames)
    n_langs          = len(lang_counts)
    cross_module_spread = len({f.split("/")[0] for f in fnames if "/" in f})
    shared_file_ratio = sum(1 for f in fnames if _SHARED_UTIL.search(f)) / max(1, len(fnames))
    boundary_density = cross_module_spread / max(1, len(fnames))

    # ── Error handling ────────────────────────────────────────────────────
    has_try_catch      = bool(_TRY_CATCH.search(patch))
    has_bare_except    = bool(_BARE_EXCEPT.search(patch))
    has_except_pass    = bool(_EXCEPT_PASS.search(patch))
    has_finally        = bool(_FINALLY.search(patch))
    has_raise_from     = bool(_RAISE_FROM.search(patch))
    has_log_in_except  = bool(_LOG_IN_EXCEPT.search(patch))
    has_reraise        = bool(_RERAISE.search(patch))

    # ── Client introductions ──────────────────────────────────────────────
    has_http_client    = bool(_HTTP_CLIENT.search(patch))
    has_db_client      = bool(_DB_CLIENT.search(patch))
    has_queue_client   = bool(_QUEUE_CLIENT.search(patch))
    has_pool_config    = bool(_DB_POOL_CONFIG.search(patch))

    # HTTP-specific: missing timeout
    http_calls = _HTTP_NO_TIMEOUT.findall(patch)
    n_http_calls = len(http_calls)
    n_http_with_timeout = len(_HTTP_TIMEOUT.findall(patch))
    http_missing_timeout = max(0, n_http_calls - n_http_with_timeout)

    # ── Resource management ───────────────────────────────────────────────
    has_context_mgr    = bool(_CONTEXT_MGR.search(patch))
    has_open_no_with   = bool(_OPEN_NO_WITH.search(patch))
    has_thread_spawn   = bool(_THREAD_SPAWN.search(patch))
    has_lock_prim      = bool(_LOCK_PRIM.search(patch))
    has_global_kw      = bool(_GLOBAL_KW.search(patch))

    # ── Security ──────────────────────────────────────────────────────────
    has_shell_true     = bool(_SUBPROCESS_SHELL.search(patch))
    has_eval_exec      = bool(_EVAL_EXEC.search(patch))
    has_pickle_loads   = bool(_PICKLE_LOADS.search(patch))
    has_hardcoded_cred = bool(_HARDCODED_CRED.search(patch))
    has_cred_in_log    = bool(_CRED_IN_LOG.search(patch))
    has_sql_fstring    = bool(_SQL_FSTRING.search(patch))
    has_input_validate = bool(_INPUT_VALIDATE.search(patch))

    # ── Observability ─────────────────────────────────────────────────────
    has_metric_emit    = bool(_METRIC_EMIT.search(patch))
    has_log_warn_err   = bool(_LOG_WARN_ERR.search(patch))
    has_health_check   = bool(_HEALTH_CHECK.search(patch))

    # Negative observability: new external client with no observability
    ext_client_no_obs  = int((has_http_client or has_db_client or has_queue_client)
                             and not has_metric_emit and not has_health_check)
    ext_client_no_log  = int((has_http_client or has_db_client)
                             and not has_log_warn_err)

    # ── Interface contracts ───────────────────────────────────────────────
    new_func_defs      = len(_NEW_FUNC_DEF.findall(patch))
    has_pub_func       = bool(_PUB_FUNC_DEF.search(patch))
    has_deprecated     = bool(_DEPRECATED.search(patch))
    has_version_suffix = bool(_VERSION_SUFFIX.search(patch))

    # ── Shared infra ──────────────────────────────────────────────────────
    has_log_fatal      = bool(_LOG_FATAL.search(patch))

    # ── Schema / data contract ────────────────────────────────────────────
    has_schema_change  = bool(_SCHEMA_CHANGE.search(patch))
    has_non_nullable   = bool(_NON_NULLABLE_COL.search(patch))

    # ── Compound risk scores (Conway review friction predictors) ──────────
    # Trust boundary score: each new external actor introduced
    trust_boundary_crossings = (
        int(imp.external > 0)
        + int(has_dep_file and imp.external > 0)
        + int(has_http_client)
        + int(has_db_client)
        + int(has_queue_client)
    )
    # Error contract quality (higher = better)
    error_contract_score = (
        + int(has_try_catch)
        + int(has_finally)
        + int(has_raise_from)
        + int(has_log_in_except)
        + int(has_reraise)
        - 2 * int(has_bare_except)
        - 3 * int(has_except_pass)
    )
    # Security risk score
    security_risk_score = (
        + 2 * int(has_shell_true)
        + 2 * int(has_sql_fstring)
        + 2 * int(has_eval_exec)
        + 2 * int(has_pickle_loads)
        + 3 * int(has_hardcoded_cred)
        +     int(has_cred_in_log)
    )
    # Operability score (higher = better)
    operability_score = (
        + 2 * int(has_metric_emit)
        + int(has_log_warn_err)
        + int(has_health_check)
        + int(has_pool_config)
        - 2 * int(ext_client_no_obs)
        - int(ext_client_no_log)
    )
    api_change_without_tests = int(
        (int(context_feats.get("ast_delta_public_defs", 0)) != 0 or has_pub_func)
        and context_feats.get("test_file_ratio", 0.0) == 0.0
    )
    schema_change_without_migration = int(has_schema_change and not has_migration)
    boundary_crossing_without_obs = int(trust_boundary_crossings > 0 and ext_client_no_obs)
    public_api_without_docs = int(
        (int(context_feats.get("ast_delta_public_defs", 0)) > 0 or has_pub_func)
        and context_feats.get("docs_file_ratio", 0.0) == 0.0
    )
    dependency_change_without_tests = int(has_dep_file and context_feats.get("test_file_ratio", 0.0) == 0.0)
    ownership_diffusion = float(
        context_feats.get("blame_unique_authors", 0.0)
        * (1.0 - context_feats.get("blame_top_author_share", 0.0))
    )
    shared_change_isolated = int(modifies_shared and cross_module_spread <= 1)
    external_io_without_safety = int(
        (has_http_client or has_db_client or has_queue_client)
        and (http_missing_timeout > 0 or ext_client_no_obs or not has_log_warn_err)
    )

    return {
        # ── metadata ──
        "primary_lang": primary_lang,
        "n_langs": n_langs,
        "n_files": len(fnames),
        "cross_module_spread": cross_module_spread,

        # ── imports (tree-sitter) ──
        "imp_total_new":    imp.total_new,
        "imp_stdlib":       imp.stdlib,
        "imp_relative":     imp.relative,
        "imp_external":     imp.external,
        "imp_ext_network":  imp.ext_network,
        "imp_ext_security": imp.ext_security,
        "imp_ext_infra":    imp.ext_infra,

        # ── file-based ──
        "has_dep_file":        int(has_dep_file),
        "has_migration_file":  int(has_migration),
        "has_config_file":     int(has_config_file),
        "has_infra_file":      int(has_infra_file),
        "has_startup_file":    int(has_startup_file),
        "modifies_shared_util":int(modifies_shared),
        "modifies_auth_code":  int(modifies_auth),
        "shared_file_ratio":   float(shared_file_ratio),
        "boundary_density":    float(boundary_density),

        # ── error handling ──
        "has_try_catch":      int(has_try_catch),
        "has_bare_except":    int(has_bare_except),
        "has_except_pass":    int(has_except_pass),
        "has_finally":        int(has_finally),
        "has_raise_from":     int(has_raise_from),
        "has_log_in_except":  int(has_log_in_except),
        "has_reraise":        int(has_reraise),

        # ── external clients ──
        "has_http_client":       int(has_http_client),
        "has_db_client":         int(has_db_client),
        "has_queue_client":      int(has_queue_client),
        "has_pool_config":       int(has_pool_config),
        "http_missing_timeout":  http_missing_timeout,
        "n_http_calls":          n_http_calls,

        # ── resource management ──
        "has_context_mgr":   int(has_context_mgr),
        "has_open_no_with":  int(has_open_no_with),
        "has_thread_spawn":  int(has_thread_spawn),
        "has_lock_prim":     int(has_lock_prim),
        "has_global_kw":     int(has_global_kw),

        # ── security ──
        "has_shell_true":     int(has_shell_true),
        "has_eval_exec":      int(has_eval_exec),
        "has_pickle_loads":   int(has_pickle_loads),
        "has_hardcoded_cred": int(has_hardcoded_cred),
        "has_cred_in_log":    int(has_cred_in_log),
        "has_sql_fstring":    int(has_sql_fstring),
        "has_input_validate": int(has_input_validate),

        # ── observability ──
        "has_metric_emit":     int(has_metric_emit),
        "has_log_warn_err":    int(has_log_warn_err),
        "has_health_check":    int(has_health_check),
        "ext_client_no_obs":   ext_client_no_obs,
        "ext_client_no_log":   ext_client_no_log,

        # ── interface contracts ──
        "new_func_defs":       new_func_defs,
        "has_pub_func":        int(has_pub_func),
        "has_deprecated":      int(has_deprecated),
        "has_version_suffix":  int(has_version_suffix),

        # ── shared infra ──
        "has_log_fatal":       int(has_log_fatal),
        "has_schema_change":   int(has_schema_change),
        "has_non_nullable_col":int(has_non_nullable),

        # ── compound scores ──
        "trust_boundary_crossings": trust_boundary_crossings,
        "error_contract_score":     error_contract_score,
        "security_risk_score":      security_risk_score,
        "operability_score":        operability_score,
        "api_change_without_tests": api_change_without_tests,
        "schema_change_without_migration": schema_change_without_migration,
        "boundary_crossing_without_obs": boundary_crossing_without_obs,
        "public_api_without_docs": public_api_without_docs,
        "dependency_change_without_tests": dependency_change_without_tests,
        "shared_change_isolated": shared_change_isolated,
        "external_io_without_safety": external_io_without_safety,
        "ownership_diffusion": ownership_diffusion,
        **context_feats,

        # ── ground-truth outcomes (for analysis) ──
        "accepted":         int(bool(pr_merged)),
        "review_friction":  int((total_review_threads or 0) >= 3
                                or (total_comments or 0) >= 8),
        "n_review_threads": int(total_review_threads or 0),
        "n_comments":       int(total_comments or 0),
    }

_write_lock = threading.Lock()

def extract_feature_worker(idx, row, fout, agg, counters, repo_dirs):
    """Worker called from ThreadPoolExecutor.
    counters = [ok, err] as a shared mutable list; mutations are under _write_lock.
    """
    (repo, iid, pr_num, base_sha, pr_merged, pr_is_draft,
    changed_files, additions, deletions,
    total_threads, total_comments,
    review_threads, patch, file_patches,
    requested_reviewers, closing_issue_id) = row

    try:
        patch_str = patch if isinstance(patch, str) else ""
        feats = extract_features(
            patch_str, file_patches,
            bool(pr_merged), int(total_threads or 0), int(total_comments or 0),
            review_threads or "",
            repo_dir=repo_dirs.get(repo),
            base_sha=base_sha or "",
        )
    except Exception as exc:
        with _write_lock:
            counters[1] += 1
            if counters[1] <= 5:
                print(f"  extract error {repo}/{iid}: {exc}", file=sys.stderr)
        return

    record = {
        "repo": repo, "instance_id": iid, "pull_number": int(pr_num or 0),
        "is_draft": int(bool(pr_is_draft)),
        "changed_files": int(changed_files or 0),
        "additions": int(additions or 0),
        "deletions": int(deletions or 0),
        **feats,
    }
    with _write_lock:
        fout.write(json.dumps(record) + "\n")
        counters[0] += 1
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                agg[k].append(float(v))
        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1} ok={counters[0]}  err={counters[1]}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",      type=int, default=30000)
    ap.add_argument("--out",        default=OUT_JSONL)
    ap.add_argument("--summary-out",default=OUT_SUMMARY)
    ap.add_argument("--min-files",  type=int, default=1)
    ap.add_argument("--max-files",  type=int, default=80)
    ap.add_argument("--workers",    type=int, default=128)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    conn = pg8000.native.Connection(
        host=cfg["ip"], port=cfg.get("port", 9999),
        user=cfg["user"], password=cfg["password"], database=cfg["database"],
    )
    rows = conn.run(
        """
        SELECT
            repo, instance_id, pull_number,
            base_sha, pr_merged, pr_is_draft,
            changed_files, additions, deletions,
            total_review_threads, total_comments,
            review_threads, patch, file_patches,
            requested_reviewers, closing_issue_id
        FROM prs_copy
        WHERE patch IS NOT NULL
          AND file_patches IS NOT NULL
          AND changed_files BETWEEN :min_f AND :max_f
          AND (COALESCE(additions,0) + COALESCE(deletions,0)) BETWEEN 5 AND 10000
        ORDER BY created_at DESC
        LIMIT :lim
        """,
        min_f=args.min_files, max_f=args.max_files, lim=args.limit,
    )
    conn.close()
    print(f"Fetched {len(rows)} rows from DB", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    repo_dirs = _repo_dir_map()
    counters = [0, 0]   # [ok, err] — mutated under _write_lock
    agg: dict[str, list] = defaultdict(list)

    with open(args.out, "w") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(extract_feature_worker, idx, row, fout, agg, counters, repo_dirs)
                    for idx, row in enumerate(rows)]
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    print(f"  unhandled worker exception: {exc}", file=sys.stderr)

    ok  = counters[0]
    err = counters[1]

    # ── Summary statistics ────────────────────────────────────────────────
    import numpy as np
    from scipy.stats import spearmanr

    acc  = [v for v in agg.get("accepted", [])]
    fric = [v for v in agg.get("review_friction", [])]
    n = len(acc)

    summary = {
        "total_rows": n,
        "errors": err,
        "accepted_rate":        float(np.mean(acc)),
        "friction_rate":        float(np.mean(fric)),
        "feature_stats": {},
        "friction_lift": {},
        "spearman_vs_accepted": {},
        "spearman_vs_friction": {},
    }

    binary_features = [k for k in agg if k not in ("accepted","review_friction","n_review_threads","n_comments")]
    acc_arr  = np.array(acc)
    fric_arr = np.array(fric)

    for feat in binary_features:
        vals = np.array(agg[feat])
        if len(vals) == 0:
            continue

        prevalence = float(np.mean(vals > 0))
        mean_val   = float(np.mean(vals))

        summary["feature_stats"][feat] = {
            "prevalence": round(prevalence, 4),
            "mean":       round(mean_val, 4),
            "p50":        round(float(np.median(vals)), 4),
            "p90":        round(float(np.percentile(vals, 90)), 4),
        }

        # Conditional friction/accept rates for binary features
        if np.max(vals) <= 1:
            pos = vals == 1
            if pos.sum() >= 10:
                pos_fric = float(fric_arr[pos].mean())
                pos_acc  = float(acc_arr[pos].mean())
                base_fric = float(fric_arr.mean())
                base_acc  = float(acc_arr.mean())
                summary["friction_lift"][feat] = {
                    "n": int(pos.sum()),
                    "friction_rate": round(pos_fric, 4),
                    "friction_lift": round(pos_fric / (base_fric + 1e-9), 3),
                    "accept_rate":   round(pos_acc, 4),
                    "accept_lift":   round(pos_acc / (base_acc + 1e-9), 3),
                }

        # Spearman correlations for all features
        if len(np.unique(vals)) > 1:
            rho_acc, _  = spearmanr(vals, acc_arr)
            rho_fric, _ = spearmanr(vals, fric_arr)
            summary["spearman_vs_accepted"][feat] = round(float(rho_acc), 4)
            summary["spearman_vs_friction"][feat] = round(float(rho_fric), 4)

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {ok} records → {args.out}")
    print(f"Summary → {args.summary_out}")

    # Quick console report
    print(f"\n{'Feature':35s}  {'%PRs':>6}  {'fric_lift':>9}  {'acc_rate':>9}  {'ρ(fric)':>8}  {'ρ(acc)':>8}")
    print("-" * 85)
    rho_fric = summary["spearman_vs_friction"]
    lift_map = summary["friction_lift"]
    sorted_feats = sorted(
        lift_map.keys(),
        key=lambda k: abs(lift_map[k]["friction_lift"] - 1.0),
        reverse=True,
    )
    for feat in sorted_feats[:30]:
        li = lift_map[feat]
        rf = rho_fric.get(feat, 0.0)
        ra = summary["spearman_vs_accepted"].get(feat, 0.0)
        print(f"  {feat:33s}  {100*li['prevalence'] if 'prevalence' in li else 100*summary['feature_stats'].get(feat,{}).get('prevalence',0):5.1f}%  "
              f"{li['friction_lift']:+.3f}x  {100*li['accept_rate']:7.1f}%  {rf:+.4f}  {ra:+.4f}")


if __name__ == "__main__":
    main()
