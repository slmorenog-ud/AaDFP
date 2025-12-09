FROM node:20-alpine as build

WORKDIR /app

# Install dependencies
COPY package.json .
RUN npm install

# Copy source
COPY . .

# Build
RUN npm run build

# Production image
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
