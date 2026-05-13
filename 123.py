FROM node:20-alpine

WORKDIR /app

COPY .npmrc ./

RUN npm config set strict-ssl false
RUN npm config set registry https://nexus.npr.nornick.ru/repository/npm-public/
RUN npm install -g pnpm@9.15.4

COPY package.json pnpm-lock.yaml* ./
RUN pnpm install

COPY . .

RUN pnpm build
