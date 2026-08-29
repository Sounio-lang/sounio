#!/usr/bin/env python3
"""
Simple MCP Server Example
This is a basic implementation that needs to be extended
"""

import json
import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mcp-server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPServer(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'healthy',
                'service': 'mcp-server',
                'version': '1.0.0'
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif parsed_path.path == '/mcp':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'protocolVersion': '2024-11-05',
                'capabilities': {
                    'tools': ['file_operations', 'code_analysis'],
                    'resources': []
                },
                'serverInfo': {
                    'name': 'Local Agent MCP Server',
                    'version': '1.0.0'
                }
            }
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        if self.path == '/mcp':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request = json.loads(post_data.decode())
                logger.info(f"Received request: {request}")
                
                # Basic MCP protocol handling
                response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id'),
                    'result': {
                        'capabilities': {
                            'tools': ['file_operations', 'code_analysis'],
                            'resources': []
                        }
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    port = 3000
    server_address = ('', port)
    httpd = HTTPServer(server_address, MCPServer)
    logger.info(f'Starting MCP server on port {port}...')
    print(f"MCP Server running at http://localhost:{port}/mcp")
    print(f"Health check: http://localhost:{port}/health")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
