# One command per thing. Run `make` to see this list.
#
# Two services now: the engine (:8100) and the backend (:8000). The backend calls
# the engine over HTTP, so the engine has to be up first.

.DEFAULT_GOAL := help
.PHONY: help setup dev engine backend tools frontend db db-stop migrate seed-db test test-py test-ui seed smoke up down logs clean

help:
	@echo ""
	@echo "  Setup (once)"
	@echo "    make setup     install dependencies and create .env"
	@echo ""
	@echo "  Run"
	@echo "    make dev       both services in this terminal (interleaved logs)"
	@echo "    make engine    AI engine only     -> http://localhost:8100/docs"
	@echo "    make backend   app backend only   -> http://localhost:8000/docs"
	@echo "    make tools     MCP tool server    -> http://localhost:8200/mcp"
	@echo "    make frontend  React UI           -> http://localhost:5173"
	@echo ""
	@echo "  Database"
	@echo "    make db        start Postgres in Docker (nothing else)"
	@echo "    make db-stop   stop it"
	@echo "    make migrate   apply Alembic migrations"
	@echo "    make seed-db   load the demo bookings and flights"
	@echo ""
	@echo "  Check"
	@echo "    make test      run all tests (python + frontend)"
	@echo "    make smoke     probe both services (needs them running)"
	@echo "    make seed      load backend/knowledge/ through the backend"
	@echo ""
	@echo "  Docker (the whole stack -- for a demo, not for developing)"
	@echo "    make up / make down / make logs"
	@echo ""

setup:
	uv sync
	@test -f .env || (cp .env.example .env && echo "created .env")
	@echo "done. next: 'make dev', then 'make smoke' in another terminal."

# Both services, one terminal. `kill 0` takes down the whole process group on
# Ctrl-C, so neither survives as an orphan holding its port.
dev:
	@trap 'kill 0' EXIT INT TERM; \
	uv run uvicorn chatbot_engine.app:app --port 8100 --reload & \
	uv run uvicorn support_agent.app:app --port 8000 --reload & \
	wait

engine:
	uv run uvicorn chatbot_engine.app:app --port 8100 --reload

backend:
	uv run uvicorn support_agent.app:app --port 8000 --reload

tools:
	uv run python -m support_agent.mcp_tools

frontend:
	cd frontend && npm install && npm run dev

# Postgres only. Everything else runs better as a plain process: `make dev`
# reloads on save, a container has to be rebuilt.
db:
	docker compose up -d postgres
	@echo "postgres on :5432 -- next: 'make migrate' then 'make seed-db'"

db-stop:
	docker compose stop postgres

migrate:
	uv run alembic -c backend/alembic.ini upgrade head

seed-db:
	uv run python backend/scripts/seed_database.py

# Both suites. The frontend's are fast and need no services, so there is no
# reason to make you remember two commands.
test:
	uv run pytest -q
	@cd frontend && npm run --silent test

test-py:
	uv run pytest -q

test-ui:
	cd frontend && npm run test

seed:
	uv run python backend/scripts/seed_knowledge.py

smoke:
	@printf 'engine   : '
	@curl -sf localhost:8100/health || (echo "DOWN -- start it with 'make engine'"; exit 1)
	@printf '\nready    : '
	@curl -sf localhost:8100/health/ready || true
	@printf '\nbackend  : '
	@curl -sf localhost:8000/health || (echo "DOWN -- start it with 'make backend'"; exit 1)
	@printf '\nchat     : '
	@curl -s -o /dev/null -w 'HTTP %{http_code} (501 until the engine has an Agent)\n' \
		-X POST localhost:8000/chat/sync \
		-H 'Content-Type: application/json' \
		-d '{"message":"what is the baggage allowance?"}'

up:
	docker compose up --build -d
	@echo "backend -> http://localhost:8000/docs   engine -> http://localhost:8100/docs"

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	rm -rf .pytest_cache frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
