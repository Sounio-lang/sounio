# Installation Guide for sounio-pkg

## Prerequisites

- Sounio compiler (`souc`) installed and in PATH
- Git (for cloning repository)
- Basic build tools (make, gcc, etc.)

## Installation Methods

### Method 1: From Source (Development)

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio-pkg.git
cd sounio-pkg

# Build the package manager
./scripts/build.sh

# Install to /usr/local/bin
sudo ./scripts/install.sh

# Verify installation
sounio-pkg --version
```

### Method 2: Using Bootstrap Script

```bash
# Download and run bootstrap script
curl -fsSL https://sounio.dev/install-sounio-pkg.sh | bash

# Or with wget
wget -qO- https://sounio.dev/install-sounio-pkg.sh | bash
```

### Method 3: Package Managers (Future)

```bash
# Ubuntu/Debian (future)
sudo apt install sounio-pkg

# macOS (future)
brew install sounio-pkg

# Windows (future)
choco install sounio-pkg
```

## Building from Source

### Step 1: Clone Repository

```bash
git clone https://github.com/sounio-lang/sounio-pkg.git
cd sounio-pkg
```

### Step 2: Build with Sounio Compiler

```bash
# Build the CLI
souc build --bin sounio-pkg --target native --optimization performance

# The binary will be at: ./target/native/sounio-pkg
```

### Step 3: Run Tests

```bash
# Run unit tests
souc test

# Run integration tests
./scripts/test-integration.sh
```

### Step 4: Install

```bash
# Install globally
sudo cp ./target/native/sounio-pkg /usr/local/bin/

# Or install locally
mkdir -p ~/.local/bin
cp ./target/native/sounio-pkg ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Development Setup

### Setting Up Development Environment

```bash
# Clone with submodules
git clone --recursive https://github.com/sounio-lang/sounio-pkg.git
cd sounio-pkg

# Install development dependencies
./scripts/setup-dev.sh

# Build in development mode
./scripts/build-dev.sh

# Run development server (for web interface)
./scripts/dev-server.sh
```

### Running in Development Mode

```bash
# Build and run directly
souc run src/main.sio -- new my-package

# Or use the development script
./scripts/dev.sh new my-package
```

## Configuration

### Environment Variables

```bash
# Registry URL (default: https://registry.sounio.dev)
export SOUNIO_REGISTRY="https://registry.sounio.dev"

# Cache directory (default: ~/.cache/sounio)
export SOUNIO_CACHE_DIR="$HOME/.cache/sounio"

# Build directory (default: ./target)
export SOUNIO_TARGET_DIR="./target"

# Log level (debug, info, warn, error)
export SOUNIO_LOG_LEVEL="info"
```

### Configuration File

Create `~/.config/sounio/config.toml`:

```toml
[registry]
url = "https://registry.sounio.dev"
cache_ttl = 3600  # seconds

[build]
default_target = "native"
default_optimization = "balanced"
parallel_jobs = 4

[network]
timeout = 30
retries = 3

[ui]
color = true
progress = true
emoji = true

[security]
verify_signatures = true
allow_insecure = false
```

## Verification

### Check Installation

```bash
# Check version
sounio-pkg --version
# Expected: sounio-pkg 0.1.0

# Check help
sounio-pkg help

# Test basic functionality
sounio-pkg new test-package --dry-run
```

### Verify Build

```bash
# Create a test package
sounio-pkg new test-package
cd test-package

# Build it
sounio-pkg build

# Run tests
sounio-pkg test

# Clean up
cd ..
rm -rf test-package
```

## Troubleshooting

### Common Issues

#### Issue: `souc` not found
```bash
# Solution: Install Sounio compiler first
# Follow instructions at https://sounio.dev/install
```

#### Issue: Permission denied
```bash
# Solution: Make script executable
chmod +x scripts/*.sh

# Or install with sudo
sudo ./scripts/install.sh
```

#### Issue: Network errors
```bash
# Solution: Check registry URL
export SOUNIO_REGISTRY="https://registry.sounio.dev"

# Or use local registry
sounio-pkg --registry http://localhost:8080
```

#### Issue: Build failures
```bash
# Solution: Check Sounio compiler version
souc --version
# Should be >= 0.5.0

# Clean and rebuild
sounio-pkg clean
sounio-pkg build --verbose
```

### Getting Help

```bash
# Show help
sounio-pkg help
sounio-pkg help <command>

# Debug mode
sounio-pkg --verbose <command>

# Check logs
cat ~/.cache/sounio/logs/sounio-pkg.log
```

## Uninstallation

```bash
# Remove binary
sudo rm /usr/local/bin/sounio-pkg

# Or remove local installation
rm ~/.local/bin/sounio-pkg

# Clean cache
rm -rf ~/.cache/sounio
rm -rf ~/.config/sounio
```

## Next Steps

After installation:

1. **Create your first package:**
   ```bash
   sounio-pkg new my-project
   cd my-project
   ```

2. **Explore examples:**
   ```bash
   sounio-pkg new --example bioinformatics
   ```

3. **Join the community:**
   - Discord: https://discord.gg/sounio
   - GitHub: https://github.com/sounio-lang
   - Documentation: https://docs.sounio.dev
```