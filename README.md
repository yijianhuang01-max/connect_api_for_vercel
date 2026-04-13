# Connect API

FastAPI adapter for the `connect_four` project.

## Endpoints

- `GET /health`
- `GET /models`
- `POST /new-game`
- `POST /move`

## Local run

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8002
```

## Vercel deployment

- Set the Vercel project Root Directory to `final_individual_project/services/connect_api`
- `app.py` is the Vercel entrypoint
- `vercel.json` contains the function configuration
- `runtime/connect_four_runtime/` packages the minimal inference runtime needed on Vercel
- `models/` includes a bundled example checkpoint and leaderboard metadata

## Notes

- The deployed Vercel bundle is intentionally lightweight and uses the bundled checkpoint in `models/`
- If you want additional selectable models on Vercel, copy more compatible checkpoints into `models/`

## Environment

- `ALLOW_ORIGINS`: comma-separated site origins allowed by CORS, or `*`
