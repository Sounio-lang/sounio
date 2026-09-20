import json
import os
from pathlib import Path
import sys
import tempfile
from jupyter_client import KernelManager
from sounio.kernel_client import KernelConnection
import sounio_kernel

assert 'site-packages' in sounio_kernel.__file__
with tempfile.TemporaryDirectory(prefix='jupyter consumer ') as directory:
    root=Path(directory)
    manager=KernelManager(kernel_name='sounio')
    manager.start_kernel(cwd=str(root))
    client=manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        connection=KernelConnection(manager,client)
        declaration=connection.execute('fn twice(x: i64) -> i64 { x + x }',timeout=60)
        assert declaration.ok, declaration.stderr
        result=connection.execute('twice(21)',timeout=60)
        assert result.ok and '42' in result.stdout,(result.status,result.stdout,result.stderr)
        bad=connection.execute('missing_symbol',timeout=60)
        assert not bad.ok,(bad.status,bad.stdout,bad.stderr)
        print(json.dumps({'status':'PASS','checks':['real_kernel_start','cell_compiles_and_executes','invalid_cell_reports_error'],'installed_kernel':sounio_kernel.__file__},indent=2))
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
