import os
import io
import sys
import time
import locale
import shlex
import signal
import socket
import subprocess
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# -------------------- paths --------------------
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.txt"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# -------------------- small utils --------------------
def is_windows() -> bool:
    return os.name == "nt"

def enable_windows_ansi():
    # 让 Windows 终端支持 ANSI 颜色（多数 Win10+ 直接 OK；这行能触发更稳）
    if is_windows():
        try:
            os.system("")
        except Exception:
            pass

ANSI = {
    "reset": "\x1b[0m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "gray": "\x1b[90m",
}

SERVICE_COLOR = {
    "lmdeploy": "cyan",
    "gptsovits": "magenta",
    "api": "green",
}

def ctext(service: str, text: str) -> str:
    color = ANSI.get(SERVICE_COLOR.get(service, "gray"), "")
    return f"{color}{text}{ANSI['reset']}"

def load_kv(path: Path) -> dict:
    cfg = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg

def venv_python(root: Path) -> Path:
    if is_windows():
        p = root / "venv" / "Scripts" / "python.exe"
    else:
        p = root / "venv" / "bin" / "python"
    if not p.exists():
        raise FileNotFoundError(f"venv python not found: {p}")
    return p

def parse_cmd(cmd: str):
    return shlex.split(cmd, posix=not is_windows())

def resolve_path_like_ps(script_dir: Path, p: str) -> Path:
    """
    仿 PS1:
    - 如果是绝对路径：原样
    - 如果是相对路径：Join-Path ScriptDir p
    - 最终 resolve 成绝对路径
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    return (script_dir / pp).resolve()

def prepend_path(env: dict, new_path: Path):
    sep = ";" if is_windows() else ":"
    env["PATH"] = str(new_path) + sep + env.get("PATH", "")

# -------------------- realtime piping --------------------
def _pump_stream(service: str, stream, is_err: bool, log_file: Path):
    prefix = f"[{service}][stderr]" if is_err else f"[{service}]"
    with open(log_file, "ab") as f:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                f.write(line)
                f.flush()
            except Exception:
                pass
            sys_enc = locale.getpreferredencoding(False)  # Windows 常见 cp936

            try:
                s = line.decode(sys_enc, errors="replace").rstrip("\n")
            except Exception:
                s = line.decode("utf-8", errors="replace").rstrip("\n")
            print(ctext(service, f"{prefix} {s}"), flush=True)

def popen_realtime(service: str, argv: list[str], cwd: Path, env: dict):
    creationflags = 0
    if is_windows():
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    p = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
        bufsize=1,  # line-buffered
    )

    out_log = LOG_DIR / f"{service}.out.log"
    err_log = LOG_DIR / f"{service}.err.log"

    t1 = threading.Thread(target=_pump_stream, args=(service, p.stdout, False, out_log), daemon=True)  # type: ignore[arg-type]
    t2 = threading.Thread(target=_pump_stream, args=(service, p.stderr, True, err_log), daemon=True)   # type: ignore[arg-type]
    t1.start()
    t2.start()

    return p

def terminate(p: subprocess.Popen, service: str, timeout=8):
    if p.poll() is not None:
        return
    try:
        if is_windows():
            try:
                p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                p.wait(timeout=timeout)
                return
            except Exception:
                pass
        else:
            p.terminate()
            p.wait(timeout=timeout)
            return
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass

# -------------------- health checks --------------------
def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "health-check"})
        with urlopen(req, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 400
    except HTTPError as e:
        # 有些服务会 404 但说明 server 已经起来了（可按需放宽）
        return e.code in (400, 401, 403, 404)
    except URLError:
        return False
    except Exception:
        return False

def wait_all_services(cfg: dict, timeout_s: int = 120):
    lm_host = cfg.get("LMDEPLOY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    lm_port = cfg.get("LMDEPLOY_PORT", cfg.get("LMDEPLOY_SERVER_PORT", "23333")).strip() or "23333"
    gpt_url = cfg.get("GPTSOVITS_URL", "http://127.0.0.1:9880").strip().rstrip("/")
    api_host = cfg.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    api_port = cfg.get("API_PORT", "8000").strip() or "8000"

    checks = {
        "lmdeploy": [f"http://{lm_host}:{lm_port}/v1/models", f"http://{lm_host}:{lm_port}/docs"],
        "gptsovits": [f"{gpt_url}/health", f"{gpt_url}/docs", f"{gpt_url}/"],
        "api": [f"http://{api_host}:{api_port}/health", f"http://{api_host}:{api_port}/docs", f"http://{api_host}:{api_port}/"],
    }

    ok = {k: False for k in checks}
    deadline = time.time() + timeout_s

    print("=========================================")
    print("[health] waiting services ready ...")
    print("=========================================")

    while time.time() < deadline:
        for name, urls in checks.items():
            if ok[name]:
                continue
            for u in urls:
                if http_ok(u, timeout=2.0):
                    ok[name] = True
                    print(ctext(name, f"[health] {name} ready: {u}"), flush=True)
                    break

        if all(ok.values()):
            print("DONE", flush=True)
            return True

        time.sleep(1)

    # 超时也把当前状态打出来
    for name, v in ok.items():
        if not v:
            print(ctext(name, f"[health][WARN] {name} not ready (timeout)"), flush=True)
    return False

# -------------------- main --------------------
def main():
    enable_windows_ansi()

    if not CONFIG_PATH.exists():
        print(f"[ERROR] config.txt not found: {CONFIG_PATH}")
        sys.exit(1)

    cfg = load_kv(CONFIG_PATH)
    py = venv_python(ROOT)

    # env 优先级：系统 env > config
    base_env = os.environ.copy()
    for k, v in cfg.items():
        base_env.setdefault(k, v)

    # -------------------- 1) LMDeploy --------------------
    lm_sub = cfg.get("LMDEPLOY_SUBCMD", "serve api_server").split()

    if not cfg.get("LMDEPLOY_MODEL_PATH", "").strip():
        print("[ERROR] LMDEPLOY_MODEL_PATH missing in config.txt")
        sys.exit(1)

    model_path = resolve_path_like_ps(ROOT, cfg["LMDEPLOY_MODEL_PATH"]).as_posix()

    lm_args = [
        str(py), "-m", "lmdeploy",
        *lm_sub,
        model_path,
        "--server-name", cfg.get("LMDEPLOY_SERVER_NAME", "0.0.0.0"),
        "--server-port", cfg.get("LMDEPLOY_SERVER_PORT", "23333"),
        "--log-level", cfg.get("LMDEPLOY_LOG_LEVEL", "WARNING"),
    ]

    def add_if(args: list[str], key: str, flag: str):
        v = cfg.get(key, "").strip()
        if v:
            args.extend([flag, v])

    add_if(lm_args, "LMDEPLOY_BACKEND", "--backend")
    add_if(lm_args, "LMDEPLOY_MODEL_FORMAT", "--model-format")
    add_if(lm_args, "LMDEPLOY_MODEL_NAME", "--model-name")
    add_if(lm_args, "LMDEPLOY_TP", "--tp")
    add_if(lm_args, "LMDEPLOY_SESSION_LEN", "--session-len")
    add_if(lm_args, "LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "--cache-max-entry-count")

    lm_env = base_env.copy()
    lm_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    lm_env.setdefault("HF_DATASETS_OFFLINE", "1")

    # cuda_stub（你要求必须加）
    cuda_stub = ROOT / "cuda_stub"
    if cuda_stub.exists():
        prepend_path(lm_env, cuda_stub)
        lm_env["CUDA_PATH"] = str(cuda_stub)
        lm_env["CUDA_HOME"] = str(cuda_stub)
        print(f"[init] lmdeploy CUDA stub: {cuda_stub}")
    else:
        print("[WARN] cuda_stub not found, LMDeploy may fail.")

    # -------------------- 2) GPT-SoVITS (完全仿 PS1) --------------------
    gpt_env = base_env.copy()

    ffmpeg_dir = ROOT / "ffmpeg" / "bin"
    if ffmpeg_dir.exists():
        prepend_path(gpt_env, ffmpeg_dir)
        print(f"[init] gptsovits ffmpeg: {ffmpeg_dir}")
    else:
        print(f"[WARN] ffmpeg not found under {ffmpeg_dir}")

    nltk_dir = ROOT / "nltk_data"
    if nltk_dir.exists():
        gpt_env["NLTK_DATA"] = str(nltk_dir)
        print(f"[init] gptsovits nltk_data: {nltk_dir}")

    ref = gpt_env.get("TTS_REF_AUDIO_PATH", "").strip()
    if ref:
        ref_abs = resolve_path_like_ps(ROOT, ref)
        if not ref_abs.exists():
            print(f"[ERROR] Reference audio not found: {ref_abs}")
            sys.exit(1)
        gpt_env["TTS_REF_AUDIO_PATH"] = str(ref_abs)
        print(f"[init] gptsovits Final TTS_REF_AUDIO_PATH: {gpt_env['TTS_REF_AUDIO_PATH']}")

    gpt_dir = ROOT / "GPT-SoVITS"
    if not gpt_dir.exists():
        print(f"[ERROR] GPT-SoVITS folder not found: {gpt_dir}")
        sys.exit(1)

    # PS1 在 GPT-SoVITS 目录内运行 api_v2.py
    gpt_args = [str(py), "api_v2.py"]

    # -------------------- 3) API Server --------------------
    api_env = base_env.copy()
    api_env["API_HOST"] = cfg.get("API_HOST", "0.0.0.0")
    api_env["API_PORT"] = cfg.get("API_PORT", "8000")
    api_env["API_RELOAD"] = cfg.get("API_RELOAD", "false")
    api_env["API_LOG_LEVEL"] = cfg.get("API_LOG_LEVEL", "info")

    api_entry = cfg.get("API_ENTRY", "api_server.py").strip() or "api_server.py"
    api_args = [str(py), api_entry]

    # -------------------- launch --------------------
    services = [
        ("lmdeploy", lm_args, ROOT, lm_env),
        ("gptsovits", gpt_args, gpt_dir, gpt_env),
        ("api", api_args, ROOT, api_env),
    ]

    print("=========================================")
    print(" Virtual Anchor - Start All (Python, PS-like)")
    print("=========================================")
    print(f"[info] root = {ROOT}")
    print(f"[info] logs = {LOG_DIR}")
    print("=========================================")

    procs: list[tuple[str, subprocess.Popen]] = []

    try:
        for i, (name, argv, cwd, env) in enumerate(services, 1):
            print(f"[{i}/3] Starting {name} ...")
            print("   ", " ".join(argv))
            p = popen_realtime(name, argv, cwd, env)
            procs.append((name, p))
            if i < 3:
                time.sleep(2)

        # 等待服务都 ready，然后输出 DONE
        timeout_s = int(cfg.get("HEALTH_CHECK_TIMEOUT_SECONDS", "120").strip() or "120")
        wait_all_services(cfg, timeout_s=timeout_s)

        # 可选：自动打开网页（保留你原逻辑）
        disable_open = cfg.get("DISABLE_AUTO_OPEN", "0").strip().lower() in ("1", "true", "yes", "on")
        if not disable_open:
            delay_ms = int(cfg.get("AUTO_OPEN_DELAY_MS", "600") or "600")
            time.sleep(max(0, delay_ms) / 1000.0)
            port = cfg.get("API_PORT", "8000")
            url = cfg.get("API_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}") + "/web/"
            try:
                import webbrowser
                webbrowser.open(url, new=1, autoraise=True)
            except Exception:
                pass

        print("=========================================")
        print(" All services launched. Press Ctrl+C to stop all.")
        print("=========================================")

        # 主循环：如果某个进程挂了就提示
        while True:
            for name, p in procs:
                code = p.poll()
                if code is not None:
                    print(ctext(name, f"[WARN] {name} exited with code {code}. See logs/"), flush=True)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[info] Ctrl+C received, stopping all...")

    finally:
        for name, p in reversed(procs):
            print(f"[stop] {name} ...")
            terminate(p, name)

        print("[info] all stopped.")

if __name__ == "__main__":
    main()