# Local PostgreSQL (Docker Desktop)

Run PostgreSQL separately from the app container.

## Start

```powershell
cd d:\master\OCRlty_cpu\infra\postgres
docker compose up -d
docker compose ps
```

## Logs

```powershell
docker compose logs -f postgres
```

## Stop

```powershell
docker compose down
```

## Stop + remove volume (wipe DB)

```powershell
docker compose down -v
```

## Quick SQL check

```powershell
docker exec -it ocrlty-postgres psql -U postgres -d ocrlty -c "select version();"
```

## App env

Use in the app `.env`:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ocrlty
```

Then run migrations from repository root:

```powershell
cd d:\master\OCRlty_cpu
alembic upgrade head
```
