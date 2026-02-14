# ---- React build stage ----
FROM node:20-alpine AS frontend
WORKDIR /app
COPY ui/frontend/package.json ui/frontend/package-lock.json ./
RUN npm install --ignore-scripts
COPY ui/frontend/ .
RUN npm run build

# ---- Runtime stage ----
FROM python:3.13-slim

# MuJoCo headless rendering (osmesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libglfw3 libglew-dev \
        patchelf gcc \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python source
COPY api/ api/
COPY controllers/ controllers/
COPY environments/ environments/
COPY config/ config/
COPY utils/ utils/
COPY models/ models/

# React build → /app/static (served by FastAPI catch-all)
COPY --from=frontend /app/build static/

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
