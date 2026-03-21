# Sounio Ecosystem Docker Setup Guide

## Files Created

### Core Docker Configuration

1. **Dockerfile** (62 lines)
   - Base: Ubuntu 24.04
   - Installs Python 3.12, pip, Jupyter Lab
   - Copies souc binary to `/usr/local/bin/souc`
   - Copies stdlib to `/opt/sounio/stdlib/`
   - Installs sounio-py (pure Python, editable)
   - Installs sounio-jupyter (editable)
   - Registers sounio Jupyter kernel
   - Copies drug-discovery examples
   - Exposes port 8888 for Jupyter Lab

2. **docker-compose.yml** (19 lines)
   - Service: sounio-lab
   - Build context: `../..` (repository root)
   - Port mapping: 8888:8888
   - Volumes: `./work` ↔ `/home/sounio/work` (persistent notebooks)
   - Environment variables:
     - `JUPYTER_TOKEN=sounio`
     - `SOUNIO_STDLIB_PATH=/opt/sounio/stdlib`
     - `SOUC=/usr/local/bin/souc`

3. **jupyter_config.py** (24 lines)
   - Authentication token: `sounio`
   - Default kernel: `sounio`
   - Allows root execution
   - Disables password auth (token-based)
   - Configures terminal and WebSocket settings

4. **README.md** (127 lines)
   - Quick start instructions
   - Directory structure
   - Environment variables
   - Common tasks (type-check, run, notebooks, terminal)
   - Troubleshooting guide
   - Further reading links

5. **.dockerignore**
   - Excludes git, cache, and build artifacts from Docker context

6. **.gitignore**
   - Excludes `work/` and `examples/` (runtime-generated)

7. **quickstart.sh** (executable)
   - Automated setup script
   - Checks Docker/Docker Compose installation
   - Creates work directory
   - Builds and starts container

8. **SETUP_GUIDE.md** (this file)
   - Installation and verification instructions

## Installation & Verification

### Prerequisites

- Docker (≥20.10)
- Docker Compose (≥2.0)
- ~2GB disk space for base image
- 4GB+ RAM allocated to Docker

### Step 1: Navigate to Docker Directory

```bash
cd /path/to/sounio/triple-sounio-ecosystem/docker/
```

### Step 2: Auto-Start with quickstart.sh

```bash
bash quickstart.sh
```

This script will:
- Verify Docker installation
- Create `work/` directory
- Build the container image
- Start Jupyter Lab
- Display access instructions

### Step 3: Manual Start (Alternative)

```bash
docker-compose up --build
```

### Step 4: Access Jupyter Lab

Open browser: `http://localhost:8888`
Token: `sounio`

## Verification Checklist

### Container Build
```bash
docker-compose build --no-cache
```
Expected: Image builds successfully, ~1.5GB final size

### Container Start
```bash
docker-compose up -d
docker-compose ps
```
Expected: `sounio-lab` shows `Up`

### Jupyter Lab Access
```bash
curl -H "Authorization: token sounio" http://localhost:8888/api/version
```
Expected: JSON response with Jupyter version

### Souc Availability
```bash
docker-compose exec sounio-lab souc --version
```
Expected: Prints version info

### Kernel Registration
```bash
docker-compose exec sounio-lab python3 -m jupyter kernelspec list
```
Expected: `sounio` kernel listed

### Example Files
```bash
docker-compose exec sounio-lab ls /home/sounio/examples/
```
Expected:
```
full_pipeline.sio
pkpd_demo.sio
screening_demo.sio
```

### Type-Check Example
```bash
docker-compose exec sounio-lab souc check /home/sounio/examples/pkpd_demo.sio
```
Expected: No errors, type summary printed

## Troubleshooting

### Build Fails: "Cannot connect to Docker daemon"
**Solution:** Start Docker daemon (Docker Desktop or `dockerd`)

### Build Fails: "No space left on device"
**Solution:** Free disk space or increase Docker's storage allocation

### Port 8888 Already in Use
**Solution:** Change port in docker-compose.yml:
```yaml
ports:
  - "8889:8888"  # New host port
```

### Jupyter Token Not Working
**Solution:** Token is `sounio` (lowercase). Check docker-compose.yml environment.

### Slow Startup (>30s)
**Normal:** JIT compilation takes 5-10s on first run. Subsequent starts are faster.

### sounio Kernel Not Available in Notebook
**Solution:** Restart Jupyter Lab:
```bash
docker-compose restart sounio-lab
```

### OOM Errors During souc Compilation
**Solution:** Allocate more RAM to Docker (Docker Dashboard → Preferences → Resources)

## Directory Layout

```
triple-sounio-ecosystem/docker/
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Service orchestration
├── jupyter_config.py            # JLab configuration
├── README.md                    # User guide
├── quickstart.sh                # Auto-start script
├── SETUP_GUIDE.md              # This file
├── .dockerignore                # Build exclusions
├── .gitignore                   # Git exclusions
├── work/                        # Persistent notebooks (created on start)
│   └── (user-created notebooks)
└── examples/                    # Symlink to examples (created on start)
    ├── pkpd_demo.sio
    ├── screening_demo.sio
    └── full_pipeline.sio
```

## Next Steps

1. **Launch container:** `docker-compose up --build`
2. **Open JLab:** http://localhost:8888 (token: `sounio`)
3. **Create notebook:** File → New → Notebook (kernel: `sounio`)
4. **Run example:** Open `examples/pkpd_demo.sio` in terminal pane
5. **Read docs:** See CLAUDE.md and docs/ in repo root

## Support Resources

- **Language Guide:** `docs/MINIMUM_VIABLE_SOUNIO.md`
- **Syntax Reference:** `docs/LLM_PROGRAMMING_GUIDE.md`
- **Project README:** `triple-sounio-ecosystem/README.md`
- **Drug-Discovery:** `triple-sounio-ecosystem/drug-discovery/`

## Advanced: Custom Configuration

### Change Jupyter Token
Edit `docker-compose.yml`:
```yaml
environment:
  JUPYTER_TOKEN: your_custom_token
```

### Change Port
Edit `docker-compose.yml`:
```yaml
ports:
  - "9999:8888"  # External port: 9999
```

### Add Custom Volumes
Edit `docker-compose.yml`:
```yaml
volumes:
  - ./work:/home/sounio/work
  - ./my_data:/home/sounio/data  # Add this
```

### Disable Token Authentication (NOT RECOMMENDED)
Edit `jupyter_config.py`:
```python
c.ServerApp.token = ''
c.ServerApp.password = ''
```

## Cleanup

### Stop Container
```bash
docker-compose down
```

### Remove Container & Image
```bash
docker-compose down --rmi all
```

### Clean All Docker Resources
```bash
docker system prune -a
```

## Performance Notes

- **First start:** 30-60s (JIT compilation)
- **Subsequent starts:** 5-10s
- **Notebook latency:** <500ms (pure Python interpreter)
- **souc compilation:** 2-5s per file (native codegen: slower)

## Version Information

Container includes:
- Ubuntu 24.04 LTS
- Python 3.12
- Jupyter Lab 4.x
- Souc v1.0.0-beta.4 (x86-64 Linux JIT)
- Sounio stdlib (current)

---

Created: 2026-03-18
Updated: 2026-03-18
