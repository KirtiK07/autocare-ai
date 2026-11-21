# syntax=docker/dockerfile:1
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_REPO_URL=""
ENV VITE_REPO_URL=$VITE_REPO_URL
RUN npm run build

# Caddy serves the SPA, reverse-proxies /api to the backend and handles HTTPS.
FROM caddy:2.10-alpine
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 80 443
