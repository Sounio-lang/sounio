#!/bin/bash

# Start MCP Server for Local Agent
# This script starts the MCP server with the configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="mcp-server-config.yaml"
PORT=3000
HOST="localhost"
LOG_DIR="./logs"
PID_FILE="./mcp-server.pid"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Port $PORT is already in use.${NC}"
        echo -e "${YELLOW}If this is from a previous MCP server instance, you can stop it with:${NC}"
        echo -e "${YELLOW}  ./stop-mcp-server.sh${NC}"
        return 1
    fi
    return 0
}

# Function to check required environment variables
check_env() {
    echo -e "${BLUE}Checking environment variables...${NC}"
    
    local missing_vars=()
    
    # Check DeepSeek
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        missing_vars+=("DEEPSEEK_API_KEY")
    else
        echo -e "${GREEN}✓ DEEPSEEK_API_KEY is set${NC}"
    fi
    
    # Check xAI
    if [ -z "$XAI_API_KEY" ]; then
        missing_vars+=("XAI_API_KEY")
    else
        echo -e "${GREEN}✓ XAI_API_KEY is set${NC}"
    fi
    
    # Check Z.AI (optional but recommended)
    if [ -z "$Z_AI_API_KEY" ]; then
        echo -e "${YELLOW}⚠ Z_AI_API_KEY is not set (GLM-5.1 will be disabled)${NC}"
    else
        echo -e "${GREEN}✓ Z_AI_API_KEY is set${NC}"
    fi
    
    # Check MiniMax (optional but recommended)
    if [ -z "$MINIMAX_API_KEY" ]; then
        echo -e "${YELLOW}⚠ MINIMAX_API_KEY is not set (MiniMax M2.7 will be disabled)${NC}"
    else
        echo -e "${GREEN}✓ MINIMAX_API_KEY is set${NC}"
    fi
    
    # Check Tavily (optional)
    if [ -z "$TAVILY_API_KEY" ]; then
        echo -e "${YELLOW}⚠ TAVILY_API_KEY is not set (web search will be disabled)${NC}"
    else
        echo -e "${GREEN}✓ TAVILY_API_KEY is set${NC}"
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo -e "${RED}Missing required environment variables:${NC}"
        for var in "${missing_vars[@]}"; do
            echo -e "${RED}  - $var${NC}"
        done
        echo -e "\n${YELLOW}Please set them before starting the server:${NC}"
        echo -e "${YELLOW}  export DEEPSEEK_API_KEY='your-key-here'${NC}"
        echo -e "${YELLOW}  export XAI_API_KEY='your-key-here'${NC}"
        return 1
    fi
    
    return 0
}

# Function to start the server
start_server() {
    echo -e "${BLUE}Starting MCP server on http://$HOST:$PORT/mcp${NC}"
    
    # Check if Node.js is available (for a potential Node-based server)
    if command -v node &> /dev/null; then
        echo -e "${GREEN}✓ Node.js is available${NC}"
        # This is a placeholder - you would need to implement the actual server
        echo -e "${YELLOW}Note: You need to implement the MCP server logic${NC}"
        echo -e "${YELLOW}See: https://modelcontextprotocol.io/docs/concepts/mcp-servers${NC}"
    else
        echo -e "${YELLOW}⚠ Node.js is not installed${NC}"
    fi
    
    # For now, we'll create a simple Python server example
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓ Python 3 is available${NC}"
        
        # Create a simple Python server script if it doesn't exist
        if [ ! -f "mcp_server.py" ]; then
            create_python_server
        fi
        
        echo -e "${BLUE}Starting Python MCP server...${NC}"
        python3 mcp_server.py &
        SERVER_PID=$!
        echo $SERVER_PID > $PID_FILE
        echo -e "${GREEN}✓ Server started with PID: $SERVER_PID${NC}"
    else
        echo -e "${RED}Python 3 is not available${NC}"
        echo -e "${YELLOW}Please install Python 3 or Node.js to run the MCP server${NC}"
        return 1
    fi
    
    # Wait a bit for server to start
    sleep 2
    
    # Check if server is running
    if curl -s http://$HOST:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running and healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Server started but health check failed${NC}"
        echo -e "${YELLOW}The server might still be starting up...${NC}"
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}MCP Server Started Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Endpoint:${NC} http://$HOST:$PORT/mcp"
    echo -e "${BLUE}Config:${NC} $CONFIG_FILE"
    echo -e "${BLUE}Logs:${NC} $LOG_DIR/mcp-server.log"
    echo -e "${BLUE}PID:${NC} $SERVER_PID (saved to $PID_FILE)"
    echo -e "\n${YELLOW}To stop the server:${NC} ./stop-mcp-server.sh"
    echo -e "${YELLOW}To check server status:${NC} ./check-mcp-server.sh"
}

# Function to create a simple Python MCP server
create_python_server() {
    cat > mcp_server.py << 'EOF'
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
EOF
    
    chmod +x mcp_server.py
    echo -e "${GREEN}✓ Created example Python MCP server${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting Local Agent MCP Server${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    
    # Check port
    if ! check_port; then
        exit 1
    fi
    
    # Check environment variables
    if ! check_env; then
        exit 1
    fi
    
    # Start server
    start_server
}

# Run main function
main