# AI Image Generator Telegram Bot

Bot: @Imagegenerattorbot
Powered by fal.ai FLUX.1 [schnell]

## Local run

    pip install -r requirements.txt
    cp .env.example .env
    # edit .env with your values
    export $(cat .env | xargs)
    python main.py

## Deploy on Render

1. Push repo to GitHub.
2. Create a **Background Worker** service, connect repo.
3. Build command: `pip install -r requirements.txt`
4. **Start command: `python main.py`**
5. Add env vars: `BOT_TOKEN`, `FAL_KEY`, `ADMIN_ID`, `DB_PATH=/data/bot_data.db`
6. Add Disk: mount `/data`, size 1 GB.
7. Deploy.

## Environment Variables

| Name | Required | Description |
|------|----------|-------------|
| BOT_TOKEN | Yes | Telegram bot token |
| FAL_KEY | Yes | fal.ai API key |
| ADMIN_ID | Yes | Admin numeric Telegram ID |
| DB_PATH | No | SQLite path (default bot_data.db) |
