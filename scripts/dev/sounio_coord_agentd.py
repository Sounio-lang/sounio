#!/usr/bin/env python3
"""Detached PTY supervisor and authenticated wake transport for Sounio agents."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import pty
import re
import secrets
import selectors
import signal
import socket
import struct
import subprocess
import sys
import termios
import time
import tty
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
RUNTIME_VERSION = "2026.08.23.2"
MAX_CONTROL_BYTES = 65536
MAX_PROMPT_BYTES = 8192
RING_BYTES = 65536
SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


class AgentdError(RuntimeError):
    pass


def slug(value: str) -> str:
    cleaned = SAFE_TOKEN.sub("-", value).strip("-")
    if not cleaned:
        raise AgentdError("identity becomes empty after normalization")
    return cleaned[:96]


def git_common_dir(cwd: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise AgentdError(f"not an attached Git worktree: {cwd}")
    return Path(result.stdout.strip()).resolve()


def state_root(cwd: Path, override: str | None) -> Path:
    value = override or os.environ.get("SOUNIO_AGENTD_DIR")
    root = Path(value).expanduser() if value else git_common_dir(cwd) / "sounio-agentd"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root.resolve()


def session_paths(root: Path, agent: str, lane: str) -> dict[str, Path]:
    directory = root / "sessions" / f"{slug(agent)}--{slug(lane)}"
    socket_root_value = os.environ.get("SOUNIO_AGENTD_SOCKET_DIR")
    if socket_root_value:
        socket_root = Path(socket_root_value).expanduser()
    else:
        runtime_root = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
        socket_root = Path(runtime_root) / f"sounio-agentd-{os.getuid()}"
    socket_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(socket_root, 0o700)
    identity = f"{root.resolve()}\0{agent}\0{lane}".encode()
    socket_name = hashlib.sha256(identity).hexdigest()[:24] + ".sock"
    return {
        "dir": directory,
        "socket": socket_root.resolve() / socket_name,
        "token": directory / "capability",
        "descriptor": directory / "session.json",
        "lock": directory / "daemon.lock",
        "start_lock": directory / "start.lock",
        "log": directory / "daemon.log",
    }


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentdError(f"cannot read session descriptor {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentdError(f"invalid session descriptor: {path}")
    return value


def process_start(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        _, separator, tail = stat.rpartition(")")
        fields = tail.split()
        if not separator or len(fields) <= 19:
            raise ValueError("short stat record")
        return fields[19]
    except (OSError, ValueError) as exc:
        raise AgentdError(f"cannot identify process {pid}: {exc}") from exc


def safe_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise AgentdError("start requires a harness command after --")
    return command


def command_argv_digest(command: list[str]) -> str:
    encoded = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_from(path: Path) -> str:
    try:
        token = path.read_text().strip()
    except OSError as exc:
        raise AgentdError(f"cannot read capability file {path}: {exc}") from exc
    if len(token) < 32:
        raise AgentdError("capability file is invalid")
    return token


def peer_uid(connection: socket.socket) -> int:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _, uid, _ = struct.unpack("3i", raw)
    return uid


def send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    connection.sendall((json.dumps(value, sort_keys=True) + "\n").encode())


def recv_line(connection: socket.socket, timeout: float = 3.0) -> tuple[bytes, bytes]:
    connection.settimeout(timeout)
    data = bytearray()
    while b"\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            raise AgentdError("agentd closed the control connection")
        data.extend(chunk)
        if len(data) > MAX_CONTROL_BYTES:
            raise AgentdError("agentd control response is too large")
    line, _, remainder = bytes(data).partition(b"\n")
    return line, remainder


def request(socket_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(3.0)
        connection.connect(str(socket_path))
        send_json(connection, payload)
        line, _ = recv_line(connection)
    except OSError as exc:
        raise AgentdError(f"cannot reach agentd socket {socket_path}: {exc}") from exc
    finally:
        connection.close()
    try:
        response = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AgentdError("agentd returned an invalid response") from exc
    if not isinstance(response, dict) or not response.get("ok"):
        reason = response.get("error", "request refused") if isinstance(response, dict) else "request refused"
        raise AgentdError(str(reason))
    return response


def direct_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.socket and args.token_file:
        return Path(args.socket).resolve(), Path(args.token_file).resolve()
    cwd = Path(args.cwd or os.getcwd()).resolve()
    paths = session_paths(state_root(cwd, args.state_dir), args.agent, args.lane)
    if paths["descriptor"].is_file():
        descriptor = read_json(paths["descriptor"])
        descriptor_socket = descriptor.get("socket")
        descriptor_token = descriptor.get("token_file")
        if isinstance(descriptor_socket, str) and isinstance(descriptor_token, str):
            return Path(descriptor_socket).resolve(), Path(descriptor_token).resolve()
    return paths["socket"], paths["token"]


def auth_payload(args: argparse.Namespace, operation: str) -> tuple[Path, dict[str, Any]]:
    socket_path, token_path = direct_paths(args)
    return socket_path, {"op": operation, "token": token_from(token_path), "protocol": PROTOCOL_VERSION}


def print_status(response: dict[str, Any]) -> None:
    ordered = (
        "state",
        "agent",
        "lane",
        "session_id",
        "worktree",
        "instance_id",
        "daemon_pid",
        "daemon_pid_start",
        "harness_pid",
        "harness_pid_start",
        "command",
        "argv_digest",
        "attached_clients",
    )
    for key in ordered:
        print(f"{key}={response.get(key, '-')}")


@dataclass
class Client:
    connection: socket.socket
    buffer: bytearray = field(default_factory=bytearray)
    attached: bool = False


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cwd = Path(args.cwd).resolve()
        self.root = state_root(self.cwd, args.state_dir)
        self.paths = session_paths(self.root, args.agent, args.lane)
        self.command = safe_command(args.command)
        self.token = token_from(self.paths["token"])
        self.instance_id = str(uuid.uuid4())
        self.selector = selectors.DefaultSelector()
        self.clients: dict[int, Client] = {}
        self.attached_fd: int | None = None
        self.ring = bytearray()
        self.stopping = False
        self.child_pid = 0
        self.master_fd = -1
        self.listener: socket.socket | None = None
        self.lock_handle: Any = None
        self.descriptor: dict[str, Any] = {}

    def setup(self) -> None:
        self.paths["dir"].mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.paths["dir"], 0o700)
        self.lock_handle = self.paths["lock"].open("a+")
        try:
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AgentdError("another supervisor generation owns this lane") from exc

        self.paths["socket"].unlink(missing_ok=True)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.paths["socket"]))
        os.chmod(self.paths["socket"], 0o600)
        self.listener.listen(16)
        self.listener.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ, "listener")

        child_pid, master_fd = pty.fork()
        if child_pid == 0:
            os.chdir(self.cwd)
            environment = os.environ.copy()
            environment.update(
                {
                    "SOUNIO_AGENTD_SOCKET": str(self.paths["socket"]),
                    "SOUNIO_AGENTD_TOKEN_FILE": str(self.paths["token"]),
                    "SOUNIO_AGENTD_AGENT": self.args.agent,
                    "SOUNIO_AGENTD_LANE": self.args.lane,
                    "SOUNIO_AGENTD_SESSION_ID": self.args.session_id,
                    "SOUNIO_AGENTD_WORKTREE": str(self.cwd),
                }
            )
            os.execvpe(self.command[0], self.command, environment)
        self.child_pid = child_pid
        self.master_fd = master_fd
        os.set_blocking(self.master_fd, False)
        self.selector.register(self.master_fd, selectors.EVENT_READ, "pty")
        daemon_pid = os.getpid()
        self.descriptor = {
            "protocol": PROTOCOL_VERSION,
            "runtime_version": RUNTIME_VERSION,
            "state": "active",
            "agent": self.args.agent,
            "lane": self.args.lane,
            "session_id": self.args.session_id,
            "worktree": str(self.cwd),
            "instance_id": self.instance_id,
            "daemon_pid": daemon_pid,
            "daemon_pid_start": process_start(daemon_pid),
            "harness_pid": self.child_pid,
            "harness_pid_start": process_start(self.child_pid),
            "command": Path(self.command[0]).name,
            "argv_digest": command_argv_digest(self.command),
            "socket": str(self.paths["socket"]),
            "token_file": str(self.paths["token"]),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(self.paths["descriptor"], self.descriptor)

    def status(self) -> dict[str, Any]:
        value = dict(self.descriptor)
        value["ok"] = True
        value["attached_clients"] = 1 if self.attached_fd is not None else 0
        return value

    def close_client(self, file_descriptor: int) -> None:
        client = self.clients.pop(file_descriptor, None)
        if client is None:
            return
        if self.attached_fd == file_descriptor:
            self.attached_fd = None
        try:
            self.selector.unregister(client.connection)
        except Exception:
            pass
        client.connection.close()

    def accept(self) -> None:
        assert self.listener is not None
        connection, _ = self.listener.accept()
        connection.setblocking(False)
        if peer_uid(connection) != os.getuid():
            connection.close()
            return
        client = Client(connection)
        self.clients[connection.fileno()] = client
        self.selector.register(connection, selectors.EVENT_READ, "client")

    def authenticate(self, request_value: dict[str, Any]) -> bool:
        candidate = request_value.get("token")
        return (
            request_value.get("protocol") == PROTOCOL_VERSION
            and isinstance(candidate, str)
            and hmac.compare_digest(candidate, self.token)
        )

    def identity_matches(self, request_value: dict[str, Any]) -> bool:
        return all(
            request_value.get(key) == self.descriptor[key]
            for key in ("agent", "lane", "session_id")
        )

    def handle_control(self, file_descriptor: int, request_value: dict[str, Any]) -> None:
        client = self.clients[file_descriptor]
        if not self.authenticate(request_value):
            send_json(client.connection, {"ok": False, "error": "authentication-refused"})
            self.close_client(file_descriptor)
            return
        operation = request_value.get("op")
        if operation == "status":
            send_json(client.connection, self.status())
            self.close_client(file_descriptor)
            return
        if operation == "wake":
            if not self.identity_matches(request_value):
                send_json(client.connection, {"ok": False, "error": "identity-mismatch"})
                self.close_client(file_descriptor)
                return
            prompt = request_value.get("prompt")
            if not isinstance(prompt, str) or not prompt or "\x00" in prompt:
                send_json(client.connection, {"ok": False, "error": "invalid-prompt"})
                self.close_client(file_descriptor)
                return
            encoded = prompt.encode()
            if len(encoded) > MAX_PROMPT_BYTES:
                send_json(client.connection, {"ok": False, "error": "prompt-too-large"})
                self.close_client(file_descriptor)
                return
            os.write(self.master_fd, encoded.rstrip(b"\r\n") + b"\r")
            send_json(
                client.connection,
                {
                    "ok": True,
                    "state": "delivered",
                    "instance_id": self.instance_id,
                    "harness_pid": self.child_pid,
                    "message_id": request_value.get("message_id", "-"),
                },
            )
            self.close_client(file_descriptor)
            return
        if operation == "attach":
            if self.attached_fd is not None and self.attached_fd != file_descriptor:
                send_json(client.connection, {"ok": False, "error": "interactive-client-active"})
                self.close_client(file_descriptor)
                return
            self.attached_fd = file_descriptor
            client.attached = True
            send_json(client.connection, {"ok": True, "state": "attached", "instance_id": self.instance_id})
            if self.ring:
                client.connection.sendall(self.ring)
            return
        if operation == "stop":
            send_json(client.connection, {"ok": True, "state": "stopping"})
            self.close_client(file_descriptor)
            self.stopping = True
            return
        send_json(client.connection, {"ok": False, "error": "unknown-operation"})
        self.close_client(file_descriptor)

    def read_client(self, file_descriptor: int) -> None:
        client = self.clients[file_descriptor]
        try:
            chunk = client.connection.recv(65536)
        except (BlockingIOError, ConnectionResetError):
            return
        if not chunk:
            self.close_client(file_descriptor)
            return
        if client.attached:
            os.write(self.master_fd, chunk)
            return
        client.buffer.extend(chunk)
        if len(client.buffer) > MAX_CONTROL_BYTES:
            send_json(client.connection, {"ok": False, "error": "control-request-too-large"})
            self.close_client(file_descriptor)
            return
        if b"\n" not in client.buffer:
            return
        line, _, remainder = bytes(client.buffer).partition(b"\n")
        if remainder:
            send_json(client.connection, {"ok": False, "error": "unexpected-control-bytes"})
            self.close_client(file_descriptor)
            return
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            send_json(client.connection, {"ok": False, "error": "invalid-control-json"})
            self.close_client(file_descriptor)
            return
        if not isinstance(value, dict):
            send_json(client.connection, {"ok": False, "error": "invalid-control-request"})
            self.close_client(file_descriptor)
            return
        self.handle_control(file_descriptor, value)

    def read_pty(self) -> None:
        try:
            chunk = os.read(self.master_fd, 65536)
        except (BlockingIOError, OSError):
            return
        if not chunk:
            self.stopping = True
            return
        self.ring.extend(chunk)
        if len(self.ring) > RING_BYTES:
            del self.ring[:-RING_BYTES]
        if self.attached_fd is None:
            return
        client = self.clients.get(self.attached_fd)
        if client is None:
            self.attached_fd = None
            return
        try:
            client.connection.sendall(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_client(self.attached_fd)

    def child_exited(self) -> tuple[bool, int]:
        try:
            pid, status = os.waitpid(self.child_pid, os.WNOHANG)
        except ChildProcessError:
            return True, 0
        if pid == 0:
            return False, 0
        return True, os.waitstatus_to_exitcode(status)

    def shutdown(self, exit_code: int | None = None) -> None:
        if exit_code is None:
            exited, exit_code = self.child_exited()
            if not exited:
                try:
                    os.killpg(self.child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    exited, exit_code = self.child_exited()
                    if exited:
                        break
                    time.sleep(0.05)
                if not exited:
                    try:
                        os.killpg(self.child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    _, status = os.waitpid(self.child_pid, 0)
                    exit_code = os.waitstatus_to_exitcode(status)
        for file_descriptor in list(self.clients):
            self.close_client(file_descriptor)
        if self.listener is not None:
            try:
                self.selector.unregister(self.listener)
            except Exception:
                pass
            self.listener.close()
        if self.master_fd >= 0:
            try:
                self.selector.unregister(self.master_fd)
            except Exception:
                pass
            os.close(self.master_fd)
        self.paths["socket"].unlink(missing_ok=True)
        if self.descriptor:
            self.descriptor["state"] = "exited"
            self.descriptor["exit_code"] = exit_code
            self.descriptor["exited_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(self.paths["descriptor"], self.descriptor)

    def run(self) -> int:
        self.setup()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stopping", True))
        exit_code: int | None = None
        try:
            while not self.stopping:
                for key, _ in self.selector.select(timeout=0.25):
                    if key.data == "listener":
                        self.accept()
                    elif key.data == "pty":
                        self.read_pty()
                    else:
                        self.read_client(key.fd)
                exited, observed = self.child_exited()
                if exited:
                    exit_code = observed
                    break
        finally:
            self.shutdown(exit_code)
        return exit_code or 0


def start_command(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    if not cwd.is_dir():
        raise AgentdError(f"worktree does not exist: {cwd}")
    root = state_root(cwd, args.state_dir)
    paths = session_paths(root, args.agent, args.lane)
    paths["dir"].mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(paths["dir"], 0o700)
    command = safe_command(args.command)
    with paths["start_lock"].open("a+") as start_lock:
        fcntl.flock(start_lock, fcntl.LOCK_EX)
        response: dict[str, Any] | None = None
        if paths["descriptor"].is_file() and paths["token"].is_file():
            try:
                descriptor = read_json(paths["descriptor"])
                descriptor_socket = Path(str(descriptor.get("socket", paths["socket"]))).resolve()
                response = request(
                    descriptor_socket,
                    {"op": "status", "token": token_from(paths["token"]), "protocol": PROTOCOL_VERSION},
                )
            except AgentdError:
                response = None
        if response is not None:
            if (
                response.get("session_id") != args.session_id
                or Path(str(response.get("worktree", ""))).resolve() != cwd
                or response.get("command") != Path(command[0]).name
                or response.get("argv_digest") != command_argv_digest(command)
            ):
                raise AgentdError(
                    "a live supervisor generation owns this agent/lane with a different identity"
                )
            print(
                "AGENTD_ALREADY_RUNNING "
                f"agent={response['agent']} lane={response['lane']} "
                f"instance_id={response['instance_id']} harness_pid={response['harness_pid']}"
            )
            return 0

        lock_probe = paths["lock"].open("a+")
        try:
            try:
                fcntl.flock(lock_probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AgentdError(
                    "a supervisor generation still owns this lane but its control channel is unavailable"
                ) from exc
            fcntl.flock(lock_probe, fcntl.LOCK_UN)
        finally:
            lock_probe.close()
        paths["socket"].unlink(missing_ok=True)
        token = secrets.token_urlsafe(48)
        atomic_write(paths["token"], (token + "\n").encode())
        log_handle = paths["log"].open("ab", buffering=0)
        serve_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_serve",
            "--agent",
            args.agent,
            "--lane",
            args.lane,
            "--session-id",
            args.session_id,
            "--cwd",
            str(cwd),
            "--state-dir",
            str(root),
            "--",
            *command,
        ]
        daemon = subprocess.Popen(
            serve_command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            close_fds=True,
        )
        log_handle.close()
        deadline = time.monotonic() + args.ready_timeout
        last_error = "supervisor did not publish its descriptor"
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                last_error = f"supervisor exited during startup with rc={daemon.returncode}"
                break
            if paths["descriptor"].is_file():
                try:
                    response = request(
                        paths["socket"],
                        {"op": "status", "token": token, "protocol": PROTOCOL_VERSION},
                    )
                    print(
                        "AGENTD_STARTED "
                        f"agent={response['agent']} lane={response['lane']} "
                        f"session_id={response['session_id']} instance_id={response['instance_id']} "
                        f"daemon_pid={response['daemon_pid']} harness_pid={response['harness_pid']} "
                        f"socket={paths['socket']} token_file={paths['token']}"
                    )
                    return 0
                except AgentdError as exc:
                    last_error = str(exc)
            time.sleep(0.05)
        try:
            tail = paths["log"].read_text(errors="replace")[-2000:]
        except OSError:
            tail = ""
        raise AgentdError(f"{last_error}{': ' + tail if tail else ''}")


def serve_command(args: argparse.Namespace) -> int:
    return Supervisor(args).run()


def status_command(args: argparse.Namespace) -> int:
    socket_path, payload = auth_payload(args, "status")
    print_status(request(socket_path, payload))
    return 0


def wake_command(args: argparse.Namespace) -> int:
    socket_path, payload = auth_payload(args, "wake")
    payload.update(
        {
            "agent": args.agent,
            "lane": args.lane,
            "session_id": args.session_id,
            "message_id": args.message_id,
            "prompt": args.prompt,
        }
    )
    response = request(socket_path, payload)
    print(
        "AGENTD_WAKE_DELIVERED "
        f"message_id={response.get('message_id', '-')} "
        f"instance_id={response['instance_id']} harness_pid={response['harness_pid']}"
    )
    return 0


def stop_command(args: argparse.Namespace) -> int:
    socket_path, payload = auth_payload(args, "stop")
    response = request(socket_path, payload)
    print(f"AGENTD_STOP state={response['state']}")
    return 0


def list_command(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    root = state_root(cwd, args.state_dir)
    descriptors = sorted((root / "sessions").glob("*/session.json"))
    count = 0
    for descriptor_path in descriptors:
        descriptor: dict[str, Any] = {}
        try:
            descriptor = read_json(descriptor_path)
            socket_path = Path(str(descriptor["socket"]))
            token_path = Path(str(descriptor["token_file"]))
            response = request(
                socket_path,
                {"op": "status", "token": token_from(token_path), "protocol": PROTOCOL_VERSION},
            )
            state = "active"
            attached = response.get("attached_clients", 0)
        except (AgentdError, KeyError):
            response = descriptor
            state = str(response.get("state", "orphaned"))
            if state == "active":
                daemon_pid = response.get("daemon_pid")
                daemon_start = response.get("daemon_pid_start")
                try:
                    state = "unresponsive" if process_start(int(daemon_pid)) == str(daemon_start) else "orphaned"
                except (AgentdError, TypeError, ValueError):
                    state = "orphaned"
            attached = 0
        print(
            "AGENTD_SESSION "
            f"state={state} agent={response.get('agent', '-')} lane={response.get('lane', '-')} "
            f"session_id={response.get('session_id', '-')} worktree={response.get('worktree', '-')} "
            f"instance_id={response.get('instance_id', '-')} harness_pid={response.get('harness_pid', '-')} "
            f"attached_clients={attached}"
        )
        count += 1
    print(f"agentd_sessions={count}")
    return 0


def attach_command(args: argparse.Namespace) -> int:
    socket_path, payload = auth_payload(args, "attach")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_terminal: list[Any] | None = None
    try:
        connection.connect(str(socket_path))
        send_json(connection, payload)
        line, remainder = recv_line(connection)
        response = json.loads(line)
        if not isinstance(response, dict) or not response.get("ok"):
            raise AgentdError(str(response.get("error", "attach refused")))
        if remainder:
            os.write(sys.stdout.fileno(), remainder)
        if sys.stdin.isatty() and not args.no_raw:
            old_terminal = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
        selector = selectors.DefaultSelector()
        connection.setblocking(False)
        selector.register(connection, selectors.EVENT_READ, "socket")
        selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")
        while True:
            for key, _ in selector.select():
                if key.data == "socket":
                    chunk = connection.recv(65536)
                    if not chunk:
                        return 0
                    os.write(sys.stdout.fileno(), chunk)
                else:
                    chunk = os.read(sys.stdin.fileno(), 65536)
                    if not chunk:
                        return 0
                    connection.sendall(chunk)
    finally:
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)
        connection.close()


def add_locator(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("--state-dir")
    parser.add_argument("--socket")
    parser.add_argument("--token-file")


def print_runtime_version(_: argparse.Namespace) -> int:
    print(f"protocol_version={PROTOCOL_VERSION}\nruntime_version={RUNTIME_VERSION}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sounio-agentd")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    version = subparsers.add_parser("runtime-version")
    version.set_defaults(function=print_runtime_version)

    start = subparsers.add_parser("start")
    start.add_argument("--agent", required=True)
    start.add_argument("--lane", required=True)
    start.add_argument("--session-id", required=True)
    start.add_argument("--cwd", required=True)
    start.add_argument("--state-dir")
    start.add_argument("--ready-timeout", type=float, default=5.0)
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(function=start_command)

    status = subparsers.add_parser("status")
    add_locator(status)
    status.set_defaults(function=status_command)

    wake = subparsers.add_parser("wake")
    add_locator(wake)
    wake.add_argument("--session-id", required=True)
    wake.add_argument("--message-id", required=True)
    wake.add_argument("--prompt", required=True)
    wake.set_defaults(function=wake_command)

    stop = subparsers.add_parser("stop")
    add_locator(stop)
    stop.set_defaults(function=stop_command)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--cwd")
    list_parser.add_argument("--state-dir")
    list_parser.set_defaults(function=list_command)

    attach = subparsers.add_parser("attach")
    add_locator(attach)
    attach.add_argument("--no-raw", action="store_true")
    attach.set_defaults(function=attach_command)
    return parser


def parse_serve_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sounio-agentd _serve", add_help=False)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(arguments)


def main() -> int:
    os.umask(0o077)
    try:
        if sys.argv[1:2] == ["_serve"]:
            return serve_command(parse_serve_args(sys.argv[2:]))
        args = build_parser().parse_args()
        return int(args.function(args))
    except (AgentdError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
