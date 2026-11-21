# Deployment (AWS EC2)

Every push to `main` runs `deploy.yml`: build both images → push to GHCR → SSH to EC2 → write `.env` from GitHub Secrets → `docker compose pull && up -d` → health check over HTTPS. The workflow skips itself until the `EC2_HOST` secret exists.

## 1. Launch the instance (one time)

1. **Check current EC2 pricing and free-tier terms first**, and create an **AWS Budget alert** (e.g. $5/month) so nothing bills by surprise.
2. EC2 → Launch instance: Ubuntu 24.04 LTS, **t3.small or larger** (PyTorch + embeddings + NLI need about 2 GB RAM), 20 GB gp3 disk.
3. Security group inbound: **22 from your IP only**, 80 and 443 from anywhere.
4. Allocate an **Elastic IP** and associate it with the instance.
5. Point a domain or free subdomain (e.g. DuckDNS) at the Elastic IP with an A record.

## 2. Prepare the server

```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit     # log in again so the group applies
mkdir -p ~/autocare-ai
```

Add a swap file on small instances to avoid out-of-memory errors during model load:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 3. GitHub Secrets (Settings → Secrets and variables → Actions)

| Secret | Value |
| --- | --- |
| `EC2_HOST` | Elastic IP or domain |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Contents of the private key (`.pem`) |
| `SITE_ADDRESS` | Your domain, e.g. `autocare.example.com` (Caddy issues HTTPS for it) |
| `LLM_PROVIDER` | `groq` or `gemini` (optional; default `groq`) |
| `LLM_API_KEY` | Groq or Gemini key |
| `POSTGRES_PASSWORD` | A long random password (letters and digits only, since it goes in a URL) |

Also create an environment called `production` (Settings → Environments). The deploy job uses it, so you can add a required reviewer if you want.

GHCR images are private by default. The workflow logs the server in with the job's token on every deploy, so nothing else is needed.

## 4. Deploy

Merge to `main` (or run the workflow manually). Then open `https://<your-domain>`.

## Roll back

Images are tagged with the short commit SHA. On the server:

```bash
cd ~/autocare-ai
sed -i 's#^BACKEND_IMAGE=.*#BACKEND_IMAGE=ghcr.io/<owner>/autocare-ai-backend:<old-sha>#; s#^FRONTEND_IMAGE=.*#FRONTEND_IMAGE=ghcr.io/<owner>/autocare-ai-frontend:<old-sha>#' .env
docker compose pull && docker compose up -d
```

Or revert the commit on `main` and let the workflow redeploy.

## Fallback: Render / Railway

The same images run unchanged. Create a managed Postgres, set `DATABASE_URL=postgresql+psycopg://...`, `LLM_API_KEY` and `CORS_ORIGINS` on the backend service, deploy `ghcr.io/<owner>/autocare-ai-backend:latest`, and serve the frontend image (or the static `frontend/dist`) with `/api` routed to the backend.
