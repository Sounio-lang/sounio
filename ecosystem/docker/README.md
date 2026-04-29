# Sounio Ecosystem Docker Container

Complete development environment for the Triple Sounio Ecosystem (L0 systems + epistemic computing language) with integrated Jupyter Lab, souc compiler, and drug-discovery examples.

## Quick Start

### Prerequisites
- Docker
- Docker Compose

### Launch the Container

```bash
cd ecosystem/docker/
docker-compose up --build
```

The container will build and start Jupyter Lab. Open your browser and navigate to:

```
http://localhost:8888
```

When prompted, enter the token: **sounio**

### What's Included

- **souc compiler** (`/usr/local/bin/souc`) — Sounio self-hosted JIT compiler
- **stdlib** (`/opt/sounio/stdlib`) — Standard library for units, epistemic types, and effects
- **sounio-py** — Pure Python Sounio interpreter (no maturin dependency)
- **sounio-jupyter** — Jupyter kernel for interactive Sounio development
- **drug-discovery examples** — Pre-loaded PK/PD modeling and screening examples
- **Jupyter Lab** — Full IDE with terminal, file browser, and notebooks

## Directory Structure

```
ecosystem/docker/
├── Dockerfile              # Ubuntu 24.04 base + Python 3.12 + Jupyter
├── docker-compose.yml      # Service definition with volume mounts
├── jupyter_config.py       # Jupyter Lab configuration
├── README.md               # This file
└── work/                   # Persistent work directory (created on first run)
    └── (user notebooks and files)
```

## Environment Variables

Inside the container:

- `SOUC=/usr/local/bin/souc` — Path to souc compiler
- `SOUNIO_STDLIB_PATH=/opt/sounio/stdlib` — Path to stdlib
- `JUPYTER_TOKEN=sounio` — Authentication token for JLab
- `PYTHONUNBUFFERED=1` — Unbuffered Python output

## Common Tasks

### Type-check a Sounio file

```bash
docker-compose exec sounio-lab souc check examples/pkpd_demo.sio
```

### Run a Sounio program

```bash
docker-compose exec sounio-lab souc run examples/screening_demo.sio
```

### Create a new notebook

In Jupyter Lab, click **File → New → Notebook**, select kernel **sounio**, and start coding:

```sio
let x = 5
let y = x + 10
```

### Access the terminal

In Jupyter Lab sidebar, click **Terminal** to open a shell with full souc access.

### Stop the container

```bash
docker-compose down
```

Data in `./work/` persists across container restarts.

## Troubleshooting

### Token not working
The default token is `sounio`. If you've set `JUPYTER_TOKEN` to something else in `docker-compose.yml`, use that instead.

### Port already in use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8889:8888"  # Maps 8889 on host to 8888 in container
```
Then access: `http://localhost:8889`

### Slow startup
JIT compilation of the souc binary takes a few seconds. Be patient on first run.

### Out of memory
Sounio programs are memory-intensive when using native compilation. Allocate more RAM to Docker:
- **Docker Desktop**: Settings → Resources → Memory
- **Docker Engine**: Edit `/etc/docker/daemon.json`

## Further Reading

- **[MINIMUM_VIABLE_SOUNIO.md](../../docs/MINIMUM_VIABLE_SOUNIO.md)** — Language basics
- **[LLM_PROGRAMMING_GUIDE.md](../../docs/LLM_PROGRAMMING_GUIDE.md)** — Syntax reference
- **[Triple Sounio Ecosystem README](../README.md)** — Project overview
- **[Drug-Discovery Guide](../drug-discovery/)** — PK/PD modeling tutorials

## Support

For issues with the Docker setup, check:
1. Docker daemon is running
2. Port 8888 is available
3. ~1GB disk space for images
4. 4GB+ RAM allocated to Docker

For Sounio language questions, see CLAUDE.md and docs/.
