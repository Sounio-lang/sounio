# Jupyter Lab Configuration for Sounio Ecosystem

# Allow root to run Jupyter
c.ServerApp.allow_root = True

# Set authentication token
c.ServerApp.token = 'sounio'

# Disable password
c.ServerApp.password = ''

# Set default kernel
c.ServerApp.default_kernel_name = 'sounio'

# Enable terminal
c.ServerApp.terminado_settings = {"shell_command": ["/bin/bash"]}

# Increase timeout for long-running computations
c.ServerApp.kernel_manager_class = 'jupyter.kernel.manager.AsyncKernelManager'
c.ServerApp.tornado_settings = {"websocket_max_message_size": 1024 * 1024 * 100}

# Lab settings
c.LabApp.notebook_dir = '/home/sounio'
c.LabApp.workspace_name = 'sounio'
