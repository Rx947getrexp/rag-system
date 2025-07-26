# 🚀 RAG 系统 - 高性能检索增强生成平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-20+-2496ED.svg)](https://www.docker.com/)

一个现代化、高性能、可扩展的 RAG (Retrieval-Augmented Generation) 系统，采用微服务架构和云原生技术栈构建。

## ✨ 核心特性

### 🎯 高性能引擎

- **Rust 核心引擎**: 极致性能的文档处理、嵌入生成和向量检索
- **异步并发**: 基于 Tokio 的高并发异步处理架构
- **零拷贝优化**: 内存安全的高效数据处理
- **SIMD 加速**: 向量计算的硬件加速支持

### 🔍 智能检索

- **混合检索策略**: 密集向量 + 稀疏关键词的最佳组合
- **多向量支持**: 文档、段落、句子多层级检索
- **实时重排序**: 基于交叉编码器的精确相关性排序
- **查询扩展**: 智能查询理解和自动扩展

### 🤖 LLM 集成

- **多提供商支持**: OpenAI、Anthropic、本地模型无缝切换
- **流式生成**: 实时响应的流式文本生成
- **上下文管理**: 智能对话历史和上下文窗口管理
- **提示工程**: 动态提示模板和优化策略

### 🔌 插件化架构

- **WASM 插件**: 安全的沙箱化插件运行环境
- **原生插件**: 高性能的 FFI 插件接口
- **热插拔**: 运行时插件加载和卸载
- **插件市场**: 丰富的社区插件生态

### 📊 企业级特性

- **多租户**: 完整的工作空间和权限管理
- **可观测性**: Prometheus + Grafana + Jaeger 全链路监控
- **高可用**: 分布式部署和自动故障恢复
- **安全性**: 端到端加密和细粒度访问控制

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (React)                          │
├─────────────────────────────────────────────────────────────────┤
│                       API 网关层 (Go)                          │
├─────────────────────────────────────────────────────────────────┤
│                      核心引擎层 (Rust)                         │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL │  Redis  │  Qdrant  │  MinIO  │  Elasticsearch   │
└─────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级           | 技术选型                      | 职责               |
| -------------- | ----------------------------- | ------------------ |
| **前端**       | React 18 + TypeScript + Vite  | 用户界面、实时交互 |
| **API 层**     | Go + Gin + gRPC               | 服务协调、认证授权 |
| **核心引擎**   | Rust + Tokio + Candle         | 文档处理、向量检索 |
| **向量数据库** | Qdrant                        | 向量存储和检索     |
| **关系数据库** | PostgreSQL                    | 元数据和用户数据   |
| **缓存**       | Redis                         | 高速缓存层         |
| **对象存储**   | MinIO / S3                    | 文件存储           |
| **搜索引擎**   | Elasticsearch                 | 全文检索           |
| **消息队列**   | RabbitMQ                      | 异步任务处理       |
| **监控**       | Prometheus + Grafana + Jaeger | 可观测性           |

## 🚀 快速开始

### 系统要求

- **操作系统**: Linux / macOS / Windows
- **内存**: 最低 8GB，推荐 16GB+
- **存储**: 最低 20GB 可用空间
- **网络**: 互联网连接（用于下载依赖和模型）

### 开发环境依赖

```bash
# 核心依赖
Docker >= 20.0
Docker Compose >= 2.0
Rust >= 1.75
Go >= 1.21
Node.js >= 18
```

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/your-org/rag-system.git
cd rag-system

# 2. 初始化环境
make init

# 3. 启动开发环境
make dev
```

启动完成后，访问以下地址：

- 🌐 **前端界面**: http://localhost:3000
- 📚 **API 文档**: http://localhost:8000/swagger/index.html
- 📊 **监控面板**: http://localhost:3001 (admin/admin123)
- 🔍 **链路追踪**: http://localhost:16686

### 手动启动（可选）

如果需要更精细的控制，可以分步启动：

```bash
# 启动基础设施
make dev-infra

# 启动 Rust 引擎
make dev-rust

# 启动 Go API
make dev-go

# 启动前端
make dev-frontend
```

## 📖 使用指南

### 基本操作

#### 1. 文档上传和索引

```bash
# 通过 API 上传文档
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf" \
  -F "title=示例文档" \
  -H "Authorization: Bearer $TOKEN"
```

#### 2. 执行检索

```bash
# 向量检索
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": "什么是人工智能？",
    "top_k": 10,
    "strategy": "hybrid"
  }'
```

#### 3. 对话交互

```bash
# 启动 RAG 对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "message": "请解释量子计算的基本原理",
    "conversation_id": "uuid",
    "enable_rag": true
  }'
```

### 高级配置

#### 自定义嵌入模型

```toml
# rag-engine/config/local.toml
[embedding.providers.local]
model_name = "sentence-transformers/all-mpnet-base-v2"
model_path = "./models/embedding"
device = "cuda"  # 或 "cpu", "mps"
batch_size = 64
```

#### 检索策略调优

```yaml
# rag-api/configs/local.yaml
retrieval:
  strategies:
    hybrid:
      dense_weight: 0.7
      sparse_weight: 0.3
    dense:
      similarity_threshold: 0.75
```

## 🛠️ 开发指南

### 项目结构

```
rag-system/
├── rag-engine/           # Rust 核心引擎
│   ├── src/
│   │   ├── embedding/    # 嵌入生成
│   │   ├── retrieval/    # 检索系统
│   │   ├── llm/         # LLM 集成
│   │   └── plugins/     # 插件系统
│   └── Cargo.toml
├── rag-api/             # Go API 网关
│   ├── internal/
│   │   ├── handlers/    # HTTP 处理器
│   │   ├── services/    # 业务逻辑
│   │   └── models/      # 数据模型
│   └── go.mod
├── rag-frontend/        # React 前端
│   ├── src/
│   │   ├── components/  # 组件库
│   │   ├── pages/       # 页面
│   │   └── services/    # API 服务
│   └── package.json
├── infrastructure/      # 基础设施
│   ├── docker/         # Docker 配置
│   ├── kubernetes/     # K8s 部署
│   └── helm/          # Helm Charts
└── Makefile           # 统一命令入口
```

### 开发工作流

```bash
# 检查代码质量
make lint

# 运行测试
make test

# 格式化代码
make format

# 构建项目
make build

# 生成文档
make docs

# 性能测试
make bench
```

### 添加新功能

1. **Rust 引擎扩展**:

    ```bash
    cd rag-engine
    cargo generate --template module
    ```

2. **Go API 端点**:

    ```bash
    cd rag-api
    # 添加新的处理器和路由
    ```

3. **前端组件**:

    ```bash
    cd rag-frontend
    npm run generate:component
    ```

## 🔧 配置参考

### 环境变量

| 变量名           | 描述                  | 默认值                   |
| ---------------- | --------------------- | ------------------------ |
| `DATABASE_URL`   | PostgreSQL 连接字符串 | `postgres://...`         |
| `REDIS_URL`      | Redis 连接字符串      | `redis://localhost:6379` |
| `QDRANT_URL`     | Qdrant 服务地址       | `http://localhost:6333`  |
| `OPENAI_API_KEY` | OpenAI API 密钥       | -                        |
| `JWT_SECRET_KEY` | JWT 签名密钥          | -                        |

### 性能调优

#### Rust 引擎优化

```toml
[concurrency]
worker_threads = 8
embedding_concurrency = 20
retrieval_concurrency = 50
llm_concurrency = 10

[cache.memory]
max_size = 2147483648  # 2GB
```

#### Go API 优化

```yaml
database:
  max_open_conns: 50
  max_idle_conns: 25
  conn_max_lifetime: 3600s

redis:
  pool_size: 100
  min_idle_conns: 20
```

## 🚀 部署指南

### Docker 部署

```bash
# 构建镜像
make docker-build

# 启动服务
docker-compose -f infrastructure/docker/docker-compose/production.yml up -d
```

### Kubernetes 部署

```bash
# 使用 Helm 部署
helm install rag-system infrastructure/helm/rag-system/ \
  --namespace rag-system \
  --create-namespace \
  --values infrastructure/helm/rag-system/values-prod.yaml
```

### 云平台部署

支持一键部署到主流云平台：

- **AWS**: EKS + RDS + ElastiCache + S3
- **Azure**: AKS + Azure Database + Redis Cache + Blob Storage  
- **GCP**: GKE + Cloud SQL + Memorystore + Cloud Storage
- **阿里云**: ACK + RDS + Redis + OSS

### 性能基准

在标准硬件配置下的性能指标：

| 指标         | 值               |
| ------------ | ---------------- |
| **检索延迟** | < 50ms (P95)     |
| **并发查询** | 10,000+ QPS      |
| **文档处理** | 1,000+ docs/min  |
| **内存使用** | < 4GB (10M 文档) |
| **索引构建** | 100K docs/hour   |

## 📊 监控和运维

### 关键指标

- **响应时间**: API 响应时间分布
- **吞吐量**: 每秒请求数和文档处理量
- **错误率**: 各组件错误率统计
- **资源使用**: CPU、内存、磁盘使用率
- **业务指标**: 检索准确率、用户满意度

### 告警规则

```yaml
# 检索延迟告警
- alert: HighSearchLatency
  expr: histogram_quantile(0.95, search_duration_seconds) > 2
  for: 5m

# 错误率告警  
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 2m
```

### 日志管理

```bash
# 查看实时日志
make logs-all

# 特定服务日志
make logs-rust    # Rust 引擎
make logs-go      # Go API
make logs-frontend # 前端服务
```

## 🔐 安全性

### 认证授权

- **JWT**: 无状态的 token 认证
- **OAuth 2.0**: 第三方登录集成
- **RBAC**: 基于角色的访问控制
- **API 密钥**: 服务间安全通信

### 数据保护

- **传输加密**: TLS 1.3 端到端加密
- **存储加密**: 静态数据 AES-256 加密
- **数据脱敏**: 敏感信息自动脱敏
- **审计日志**: 完整的操作审计追踪

### 安全最佳实践

- 定期安全扫描和漏洞修复
- 最小权限原则和网络隔离
- 容器镜像安全基线检查
- 依赖项安全性持续监控

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献流程

1. **Fork 项目** 到您的 GitHub 账户
2. **创建特性分支**: `git checkout -b feature/amazing-feature`
3. **提交更改**: `git commit -m 'Add amazing feature'`
4. **推送分支**: `git push origin feature/amazing-feature`
5. **创建 Pull Request**

### 开发规范

- **代码风格**: 遵循各语言的官方代码规范
- **测试覆盖**: 新功能必须包含单元测试
- **文档更新**: 重要变更需要更新相关文档
- **性能测试**: 性能相关改动需要提供基准测试

### 问题报告

使用 [GitHub Issues](https://github.com/your-org/rag-system/