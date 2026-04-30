# LawCopilot Frontend - React + Ant Design
FROM node:20-alpine AS build

WORKDIR /app

# 复制 package 文件
COPY package.json package-lock.json* ./

# 安装依赖（使用国内镜像）
RUN npm install --registry=https://registry.npmmirror.com

# 复制源码
COPY . .

# 构建生产版本
RUN npm run build

# ===== 生产阶段：Nginx 托管静态文件 =====
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
