# EDVORA API

AI-powered academic roadmap and chat assistant using Gemini. Generates 5-step study plans and answers course-related questions in Arabic.

## Endpoints

- **GET** `/health` — health check
- **GET** `/catalog/courses` — list all courses in catalog
- **POST** `/roadmap-lite` — generate study roadmap
- **POST** `/chat` — ask course-related questions

## Deploy on Render

1. Push this repo to GitHub
2. Render → New → Web Service → connect repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Add env var: `GEMINI_API_KEY`
6. Deploy
