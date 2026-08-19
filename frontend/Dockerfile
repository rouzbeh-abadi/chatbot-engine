# Two stages: build the bundle with Node, then serve the static files with nginx.
# The Node toolchain never ships -- the final image is nginx plus a few hundred KB
# of assets.

FROM node:24-alpine AS build
WORKDIR /app

# Copy the manifests first so a source edit does not invalidate the install layer.
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
