FROM node:20-alpine

WORKDIR /app

RUN npm config set strict-ssl false
RUN npm install -g pnpm

COPY package.json pnpm-lock.yaml* ./
RUN pnpm install

COPY . .

RUN pnpm build
