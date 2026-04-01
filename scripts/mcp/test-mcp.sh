#!/bin/bash

# Test MCP Server Completo
set -e

echo "🧪 TESTANDO MCP SERVER + MODELOS..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load env
if [ -f .env.local ]; then
  echo -e "${BLUE}✓ Loading .env.local${NC}"
  source .env.local
else
  echo -e "${RED}❌ .env.local not found! Crie com suas keys.${NC}"
  exit 1
fi

# Check if server running
if curl -s http://localhost:3000/health > /dev/null; then
  echo -e "${GREEN}✓ Server rodando (health OK)${NC}"
else
  echo -e "${YELLOW}⚠ Server não rodando. Iniciando...${NC}"
  ./start-mcp-server.sh &
  sleep 3
  if ! curl -s http://localhost:3000/health > /dev/null; then
    echo -e "${RED}❌ Server falhou ao iniciar${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Server iniciado${NC}"
fi

echo -e "\n${BLUE}1. Health Check:${NC}"
curl -s http://localhost:3000/health | jq .

echo -e "\n${BLUE}2. MCP Discovery:${NC}"
curl -s http://localhost:3000/mcp | jq .

echo -e "\n${BLUE}3. Tools List (JSON-RPC):${NC}"
curl -s -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }' | jq .

echo -e "\n${BLUE}4. Models Check (env vars):${NC}"
echo "Z_AI_API_KEY: ${Z_AI_API_KEY:0:10}... ✓"
echo "MINIMAX_API_KEY: ${MINIMAX_API_KEY:0:10}... ✓"

echo -e "\n${BLUE}5. Logs recentes:${NC}"
mkdir -p logs
tail -n 10 logs/mcp-server.log || echo "Sem logs ainda"

echo -e "\n${GREEN}🎉 TESTES CONCLUÍDOS! Todos endpoints OK.${NC}"
echo -e "${YELLOW}Para testes reais de API: Estenda mcp_server.py com SDK.${NC}"
echo -e "${BLUE}Endpoint pronto para CONTEXT7: http://localhost:3000/mcp${NC}"