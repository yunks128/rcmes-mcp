# RCMES Web UI

React frontend for the RCMES Climate Data Explorer.

## Quick Start

### Option 1: Development (two terminals)

**Terminal 1 - Start the API server:**
```bash
pip install -e .
rcmes-api
# Or: uvicorn rcmes_mcp.api:app --reload --port 8502
```

**Terminal 2 - Start the React dev server:**
```bash
cd web
npm install
npm run dev
```

Access: http://localhost:5173 (proxies API to :8502)

### Option 2: Production (single server on port 8502)

```bash
# Build the React app
cd web
npm install
npm run build

# Start the API server (serves both API and static files)
cd ..
rcmes-api
```

Access: http://34.31.165.25:8502

## Development

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Architecture

```
web/
├── src/
│   ├── main.tsx       # Entry point
│   ├── App.tsx        # Main application component
│   ├── api.ts         # API client functions
│   └── index.css      # Global styles
├── public/
│   └── favicon.svg    # App icon
└── index.html         # HTML template
```

The frontend communicates with the FastAPI backend (`src/rcmes_mcp/api.py`) which wraps the RCMES climate analysis tools.

## API Endpoints

The backend provides these REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available climate models |
| `/api/variables` | GET | List available variables |
| `/api/scenarios` | GET | List emissions scenarios |
| `/api/datasets` | GET | List loaded datasets |
| `/api/load-data` | POST | Load climate data from S3 |
| `/api/analyze` | POST | Run analysis on dataset |
| `/api/visualize` | POST | Generate visualization |
| `/api/health` | GET | Health check |
