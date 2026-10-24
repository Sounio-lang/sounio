"""Paired connectivity observation; never accesses authenticated API resources."""
import argparse, datetime, json, os, socket, time, urllib.request
p=argparse.ArgumentParser()
p.add_argument("--role", choices=("control","isolated"), required=True)
args=p.parse_args()
target=os.environ["KUBERNETES_SERVICE_HOST"]
port=int(os.environ["KUBERNETES_SERVICE_PORT"])
public=False
public_error=None
try:
    with urllib.request.urlopen("https://github.com", timeout=15) as r:
        public = r.status == 200
except Exception as exc:
    public_error=type(exc).__name__
start=time.monotonic()
private=False
private_error=None
try:
    with socket.create_connection((target,port),timeout=5):
        private=True
except Exception as exc:
    private_error=type(exc).__name__
result=dict(schema="pireus-runner-network-observation-v1",role=args.role,
    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    public_url="https://github.com",public_https_ok=public,public_error=public_error,
    private_host=target,private_port=port,private_tcp_connected=private,
    private_error=private_error,private_probe_seconds=time.monotonic()-start,
    authenticated_api_request=False,uid=os.getuid())
print("PIREUS_NETWORK_PROBE "+json.dumps(result),flush=True)
expected = private if args.role=="control" else (not private and private_error=="TimeoutError")
raise SystemExit(0 if public and expected and os.getuid()==1000 else 1)
