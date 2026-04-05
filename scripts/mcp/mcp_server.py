#!/usr/bin/env python3
&quot;&quot;&quot;
Production MCP Server with Tavily Web Search &amp; Model Routing
- Loads .env.local for API keys
- Parses mcp-server-config.yaml
- MCP endpoints: /mcp (discovery), /health, /chat/completions (model proxy + routing), /tools/web_search
- Tools: Tavily search (structured results)
- Models: DeepSeek, Grok variants, GLM-5.1, MiniMax M2.7
- Logging to ./logs/mcp-server.log
&quot;&quot;&quot;

import json
import logging
import os
import yaml
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import requests
from typing import Dict, Any, List, Optional

# Load environment from .env.local
load_dotenv('.env.local', override=True)

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mcp-server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
with open('mcp-server-config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

MODELS = {m['name']: m for m in CONFIG['models']}
TOOLS = {
    'web_search': {
        'description': 'Search the web using Tavily for up-to-date information',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Search query'}
            },
            'required': ['query']
        }
    },
    # Add Sounio tools later
    'sounio_compiler': {
        'description': 'Run Sounio compiler',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {'type': 'string'}
            },
            'required': ['command']
        }
    }
}

def call_tavily(query: str) -> Dict[str, Any]:
    &quot;&quot;&quot;Tavily Search API&quot;&quot;&quot;
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        raise ValueError('TAVILY_API_KEY not set')
    
    url = 'https://api.tavily.com/search'
    payload = {
        'api_key': api_key,
        'query': query,
        'search_depth': 'basic',
        'include_answer': True,
        'max_results': 5
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def proxy_model_request(model_name: str, messages: List[Dict], tools: Optional[List] = None, **kwargs) -> Dict:
    &quot;&quot;&quot;Proxy chat to model provider&quot;&quot;&quot;
    model_config = MODELS.get(model_name)
    if not model_config:
        raise ValueError(f'Model {model_name} not found')
    
    url = model_config.get('endpoint', f'https://api.{model_config["provider"]}.com/v1/chat/completions')
    headers = {
        'Authorization': f'Bearer {os.getenv(model_config["provider"].upper().replace("-", "_") + "_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model_config['model'],
        'messages': messages,
        **kwargs
    }
    if tools:
        payload['tools'] = tools
    
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

class MCPServer(BaseHTTPRequestHandler):
    def send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            self.send_json({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'models': list(MODELS.keys()),
                'tools': list(TOOLS.keys())
            })
        elif path == '/mcp':
            # MCP Discovery
            self.send_json({
                'protocolVersion': '2024-11-05',
                'capabilities': {
                    'models': True,
                    'tools': list(TOOLS.keys()),
                    'resources': []
                },
                'tools': [
                    {
                        'name': name,
                        'description': tool['description'],
                        'inputSchema': tool['parameters']
                    }
                    for name, tool in TOOLS.items()
                ],
                'serverInfo': {
                    'name': CONFIG['name'],
                    'version': CONFIG['version']
                }
            })
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Type-Length'] if 'Content-Type-Length' in self.headers else self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode())
        
        path = urlparse(self.path).path
        
        if path == '/mcp/chat/completions':
            # Model chat proxy + routing
            model_name = request.get('model', CONFIG['routing']['default'])
            messages = request['messages']
            
            # Simple routing based on config patterns (extend as needed)
            user_msg = messages[-1]['content'].lower() if messages else ''
            for rule in CONFIG['routing']['rules']:
                if rule['pattern'] in user_msg:
                    model_name = rule['model']
                    break
            
            try:
                response = proxy_model_request(model_name, messages, tools=list(TOOLS.values()))
                self.send_json(response)
            except Exception as e:
                logger.error(f'Model error: {e}')
                self.send_json({'error': str(e)}, 500)
        
        elif path.startswith('/tools/'):
            tool_name = path.split('/')[-1]
            if tool_name == 'web_search':
                query = request.get('query')
                if not query:
                    self.send_json({'error': 'Missing query'}, 400)
                    return
                try:
                    result = call_tavily(query)
                    self.send_json({
                        'tool': 'web_search',
                        'result': result,
                        'query': query
                    })
                except Exception as e:
                    logger.error(f'Tavily error: {e}')
                    self.send_json({'error': str(e)}, 500)
            elif tool_name == 'sounio_compiler':
                # Stub - integrate resolve_souc.sh etc.
                command = request.get('command', 'help')
                self.send_json({
                    'tool': 'sounio_compiler',
                    'output': f'Sounio command: {command} (stub - integrate scripts/lib/resolve_souc.sh)'
                })
            else:
                self.send_json({'error': f'Unknown tool: {tool_name}'}, 404)
        
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    port = CONFIG['server']['port']
    httpd = HTTPServer(('localhost', port), MCPServer)
    logger.info(f'MCP Server running at http://localhost:{port}')
    logger.info(f'Discovery: http://localhost:{port}/mcp')
    logger.info(f'Health: http://localhost:{port}/health')
    logger.info(f'Tools: POST /tools/web_search {"query": "test"}')
    httpd.serve_forever()
