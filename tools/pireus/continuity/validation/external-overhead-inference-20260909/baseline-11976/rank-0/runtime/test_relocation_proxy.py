#!/usr/bin/env python3
"""Real loopback transport controls for the pre-authority activation gate."""
import asyncio, contextlib, errno, pathlib, socket, sys, tempfile, unittest
class ProxyControls(unittest.IsolatedAsyncioTestCase):
    async def test_activation_gate_and_half_close(self):
        connections=[]
        async def echo(reader,writer):
            connections.append(1)
            data=await reader.read()
            writer.write(data);await writer.drain();writer.close();await writer.wait_closed()
        target=await asyncio.start_server(echo,"127.0.0.1",0)
        target_port=target.sockets[0].getsockname()[1]
        with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            marker=pathlib.Path(directory)/"active"
            proc=await asyncio.create_subprocess_exec(sys.executable,str(pathlib.Path(__file__).with_name("relocation_proxy.py")),
                "--listen-host","127.0.0.1","--listen-port",str(port),"--target-host","127.0.0.1","--target-port",str(target_port),
                "--activation-file",str(marker),"--activation-value","test-target",
                "--maintenance-client","127.0.0.1",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            async def exchange(source_ip):
                r,w=await asyncio.open_connection("127.0.0.1",port,local_addr=(source_ip,0))
                try:
                    w.write(b"synthetic-proxy-control");await w.drain();w.write_eof()
                    return await asyncio.wait_for(r.read(),3)
                except OSError as e:
                    if e.errno in [errno.ECONNRESET,errno.EPIPE,errno.ENOTCONN]:return b""
                    raise
                finally:
                    w.close()
                    with contextlib.suppress(OSError):await w.wait_closed()
            try:
                self.assertEqual(await asyncio.wait_for(proc.stdout.readline(),5),b"POSTGRES_FORWARDER_READY\n")
                self.assertEqual(await exchange("127.0.0.1"),b"synthetic-proxy-control")
                self.assertEqual(len(connections),1)
                self.assertEqual(await exchange("127.0.0.2"),b"")
                self.assertEqual(len(connections),1)
                marker.write_text("wrong-authority")
                self.assertEqual(await exchange("127.0.0.2"),b"");self.assertEqual(len(connections),1)
                marker.write_text("test-target\n")
                self.assertEqual(await exchange("127.0.0.2"),b"synthetic-proxy-control")
                self.assertEqual(len(connections),2)
                marker.unlink()
                self.assertEqual(await exchange("127.0.0.2"),b"");self.assertEqual(len(connections),2)
            finally:
                proc.terminate();await asyncio.wait_for(proc.wait(),5)
                target.close();await target.wait_closed()
            self.assertEqual(proc.returncode,0)
            self.assertEqual(await proc.stderr.read(),b"")
if __name__=="__main__":unittest.main()
