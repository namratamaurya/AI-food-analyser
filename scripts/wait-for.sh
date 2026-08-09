#!/usr/bin/env bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 host port -- command"
  echo "Example: $0 mongo 27017 -- uvicorn app.api:app --host 0.0.0.0 --port 8000"
  exit 1
fi

host="$1"
port="$2"
shift 2

# wait for host:port to be available
python - <<PY
import socket,sys,time
host=sys.argv[1]
port=int(sys.argv[2])
for _ in range(60):
    try:
        s=socket.create_connection((host,port),2)
        s.close()
        print(f"{host}:{port} is available")
        sys.exit(0)
    except Exception:
        time.sleep(1)
print(f"Timed out waiting for {host}:{port}")
sys.exit(1)
PY "$host" "$port"

# exec the remaining command
exec "$@"
