#!/bin/bash

# RAG 系统开发环境启动脚本
# scripts/dev-start.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    # 检查 Rust
    if ! command -v cargo &> /dev/null; then
        log_error "Rust 未安装，请先安装 Rust"
        exit 1
    fi

    # 检查 Go
    if ! command -v go &> /dev/null; then
        log_error "Go 未安装，请先安装 Go"
        exit 1
    fi

    # 检查 Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js 未安装，请先安装 Node.js"
        exit 1
    fi

    log_success "所有依赖检查通过"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p data/{postgres,redis,qdrant,minio,elasticsearch,grafana,prometheus,rabbitmq}
    mkdir -p logs
    mkdir -p uploads
    mkdir -p models/{embedding,llm}
    mkdir -p plugins

    log_success "目录创建完成"
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."

    # 复制环境变量模板
    if [ ! -f .env ]; then
        cat > .env << 'EOF'
# RAG 系统环境变量

# 数据库配置
DATABASE_URL=postgres://rag_user:rag_password@localhost:5432/rag_development
REDIS_URL=redis://localhost:6379/0

# RAG 引擎配置
RAG_ENGINE_GRPC_URL=localhost:9090
QDRANT_URL=http://localhost:6333
ELASTICSEARCH_URL=http://localhost:9200

# MinIO 配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# 认证配置
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production

# 监控配置
JAEGER_ENDPOINT=http://localhost:14268/api/traces

# 开发模式
RUST_LOG=debug
GIN_MODE=debug

# API 密钥 (开发时可以为空)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_KEY=
EOF
        log_success "环境变量文件 .env 已创建"
    else
        log_info "环境变量文件 .env 已存在，跳过创建"
    fi

    # 加载环境变量
    source .env
}

# 启动基础设施服务
start_infrastructure() {
    log_info "启动基础设施服务..."

    cd infrastructure/docker/docker-compose

    # 启动基础设施
    docker-compose -f development.yml up -d postgres redis qdrant minio elasticsearch jaeger prometheus grafana rabbitmq

    log_info "等待服务启动..."
    sleep 10

    # 检查服务状态
    check_service_health

    cd ../../..
}

# 检查服务健康状态
check_service_health() {
    log_info "检查服务健康状态..."

    # 检查 PostgreSQL
    if docker-compose -f development.yml exec -T postgres pg_isready -U rag_user > /dev/null 2>&1; then
        log_success "PostgreSQL 服务正常"
    else
        log_warning "PostgreSQL 服务可能还未完全启动"
    fi

    # 检查 Redis
    if docker-compose -f development.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis 服务正常"
    else
        log_warning "Redis 服务可能还未完全启动"
    fi

    # 检查 Qdrant
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        log_success "Qdrant 服务正常"
    else
        log_warning "Qdrant 服务可能还未完全启动"
    fi

    # 检查 MinIO
    if curl -s http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        log_success "MinIO 服务正常"
    else
        log_warning "MinIO 服务可能还未完全启动"
    fi
}

# 初始化数据库
init_database() {
    log_info "初始化数据库..."

    # 等待 PostgreSQL 完全启动
    log_info "等待 PostgreSQL 完全启动..."
    sleep 5

    # 运行数据库迁移 (如果有的话)
    # cd rag-api && go run cmd/migrate/main.go up && cd ..

    log_success "数据库初始化完成"
}

# 构建并启动 Rust 引擎
start_rust_engine() {
    log_info "构建并启动 Rust 引擎..."

    cd rag-engine

    # 检查配置文件
    if [ ! -f config/local.toml ]; then
        log_info "创建 Rust 引擎配置文件..."
        cp ../rust_config.toml config/local.toml
    fi

    # 构建项目 (Debug 模式)
    log_info "构建 Rust 引擎 (Debug 模式)..."
    cargo build

    # 启动服务 (后台运行)
    log_info "启动 Rust 引擎服务..."
    RUST_LOG=debug cargo run --bin rag-engine-server > ../logs/rag-engine.log 2>&1 &
    RUST_PID=$!
    echo $RUST_PID > ../logs/rag-engine.pid

    cd ..

    # 等待服务启动
    log_info "等待 Rust 引擎启动..."
    sleep 10

    # 检查服务是否启动成功
    if curl -s http://localhost:8081/health > /dev/null 2>&1; then
        log_success "Rust 引擎启动成功 (PID: $RUST_PID)"
    else
        log_warning "Rust 引擎可能还未完全启动，请检查日志: logs/rag-engine.log"
    fi
}

# 启动 Go API 服务
start_go_api() {
    log_info "启动 Go API 服务..."

    cd rag-api

    # 检查配置文件
    if [ ! -f configs/local.yaml ]; then
        log_info "创建 Go API 配置文件..."
        cp ../go_config.yaml configs/local.yaml
    fi

    # 下载依赖
    log_info "下载 Go 依赖..."
    go mod download

    # 启动服务 (后台运行)
    log_info "启动 Go API 服务..."
    go run cmd/server/main.go -config configs/local.yaml > ../logs/rag-api.log 2>&1 &
    GO_PID=$!
    echo $GO_PID > ../logs/rag-api.pid

    cd ..

    # 等待服务启动
    log_info "等待 Go API 服务启动..."
    sleep 5

    # 检查服务是否启动成功
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Go API 服务启动成功 (PID: $GO_PID)"
    else
        log_warning "Go API 服务可能还未完全启动，请检查日志: logs/rag-api.log"
    fi
}

# 启动前端服务
start_frontend() {
    log_info "启动前端服务..."

    cd rag-frontend

    # 安装依赖
    if [ ! -d node_modules ]; then
        log_info "安装前端依赖..."
        npm install
    fi

    # 启动开发服务器 (后台运行)
    log_info "启动前端开发服务器..."
    npm run dev > ../logs/rag-frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/rag-frontend.pid

    cd ..

    # 等待服务启动
    log_info "等待前端服务启动..."
    sleep 10

    # 检查服务是否启动成功
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        log_success "前端服务启动成功 (PID: $FRONTEND_PID)"
    else
        log_warning "前端服务可能还未完全启动，请检查日志: logs/rag-frontend.log"
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态概览:"
    echo
    echo "🗄️  基础设施服务:"
    echo "   • PostgreSQL:     http://localhost:5432"
    echo "   • Redis:          http://localhost:6379"
    echo "   • Qdrant:         http://localhost:6333"
    echo "   • MinIO:          http://localhost:9000 (admin/admin123)"
    echo "   • Elasticsearch:  http://localhost:9200"
    echo "   • Jaeger:         http://localhost:16686"
    echo "   • Prometheus:     http://localhost:9090"
    echo "   • Grafana:        http://localhost:3001 (admin/admin123)"
    echo "   • RabbitMQ:       http://localhost:15672 (rag_user/rag_password)"
    echo
    echo "🚀 应用服务:"
    echo "   • Rust 引擎:      http://localhost:8080 (HTTP), :9090 (gRPC)"
    echo "   • Go API:         http://localhost:8000"
    echo "   • 前端界面:       http://localhost:3000"
    echo
    echo "📊 监控服务:"
    echo "   • 健康检查:       http://localhost:8000/health"
    echo "   • API 文档:       http://localhost:8000/swagger/index.html"
    echo "   • 指标收集:       http://localhost:8002/metrics"
    echo
    echo "📋 日志文件:"
    echo "   • Rust 引擎:      logs/rag-engine.log"
    echo "   • Go API:         logs/rag-api.log"
    echo "   • 前端服务:       logs/rag-frontend.log"
    echo
}

# 创建停止脚本
create_stop_script() {
    cat > scripts/dev-stop.sh << 'EOF'
#!/bin/bash

# RAG 系统开发环境停止脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_info "🛑 停止 RAG 系统开发环境..."

# 停止应用服务
if [ -f logs/rag-engine.pid ]; then
    RUST_PID=$(cat logs/rag-engine.pid)
    if kill -0 $RUST_PID 2>/dev/null; then
        kill $RUST_PID
        log_info "Rust 引擎已停止 (PID: $RUST_PID)"
    fi
    rm -f logs/rag-engine.pid
fi

if [ -f logs/rag-api.pid ]; then
    GO_PID=$(cat logs/rag-api.pid)
    if kill -0 $GO_PID 2>/dev/null; then
        kill $GO_PID
        log_info "Go API 服务已停止 (PID: $GO_PID)"
    fi
    rm -f logs/rag-api.pid
fi

if [ -f logs/rag-frontend.pid ]; then
    FRONTEND_PID=$(cat logs/rag-frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        log_info "前端服务已停止 (PID: $FRONTEND_PID)"
    fi
    rm -f logs/rag-frontend.pid
fi

# 停止基础设施服务
log_info "停止基础设施服务..."
cd infrastructure/docker/docker-compose
docker-compose -f development.yml down

log_success "✅ RAG 系统开发环境已完全停止"
EOF

    chmod +x scripts/dev-stop.sh
    log_success "停止脚本已创建: scripts/dev-stop.sh"
}

# 创建重启脚本
create_restart_script() {
    cat > scripts/dev-restart.sh << 'EOF'
#!/bin/bash

# RAG 系统开发环境重启脚本

echo "🔄 重启 RAG 系统开发环境..."

# 停止所有服务
./scripts/dev-stop.sh

# 等待一会儿
sleep 3

# 重新启动
./scripts/dev-start.sh
EOF

    chmod +x scripts/dev-restart.sh
    log_success "重启脚本已创建: scripts/dev-restart.sh"
}

# 创建日志查看脚本
create_logs_script() {
    cat > scripts/dev-logs.sh << 'EOF'
#!/bin/bash

# RAG 系统日志查看脚本

SERVICE=${1:-all}

case $SERVICE in
    "rust"|"engine")
        echo "📋 Rust 引擎日志:"
        tail -f logs/rag-engine.log
        ;;
    "go"|"api")
        echo "📋 Go API 日志:"
        tail -f logs/rag-api.log
        ;;
    "frontend"|"react")
        echo "📋 前端服务日志:"
        tail -f logs/rag-frontend.log
        ;;
    "all")
        echo "📋 所有服务日志 (实时):"
        echo "使用 Ctrl+C 退出"
        echo "========================"
        tail -f logs/*.log
        ;;
    *)
        echo "用法: $0 [rust|go|frontend|all]"
        echo "默认显示所有日志"
        ;;
esac
EOF

    chmod +x scripts/dev-logs.sh
    log_success "日志查看脚本已创建: scripts/dev-logs.sh"
}

# 主函数
main() {
    echo "🚀 RAG 系统开发环境启动脚本"
    echo "=============================="

    # 创建 scripts 目录
    mkdir -p scripts logs

    # 检查依赖
    check_dependencies

    # 创建目录
    create_directories

    # 设置环境变量
    setup_environment

    # 启动基础设施
    start_infrastructure

    # 初始化数据库
    init_database

    # 启动 Rust 引擎
    start_rust_engine

    # 启动 Go API
    start_go_api

    # 启动前端
    start_frontend

    # 创建管理脚本
    create_stop_script
    create_restart_script
    create_logs_script

    # 显示状态
    show_status

    log_success "🎉 RAG 系统开发环境启动完成!"
    echo
    log_info "💡 管理命令:"
    echo "   • 停止服务:     ./scripts/dev-stop.sh"
    echo "   • 重启服务:     ./scripts/dev-restart.sh"
    echo "   • 查看日志:     ./scripts/dev-logs.sh [rust|go|frontend|all]"
    echo
    log_info "🔗 快速访问:"
    echo "   • 前端界面:     http://localhost:3000"
    echo "   • API 文档:     http://localhost:8000/swagger/index.html"
    echo "   • 监控面板:     http://localhost:3001 (Grafana)"
    echo "   • 链路追踪:     http://localhost:16686 (Jaeger)"
    echo
}

# 错误处理
trap 'log_error "脚本执行失败，请检查错误信息"; exit 1' ERR

# 检查是否在项目根目录
if [ ! -f "README.md" ] || [ ! -d "rag-engine" ] || [ ! -d "rag-api" ] || [ ! -d "rag-frontend" ]; then
    log_error "请在项目根目录运行此脚本"
    exit 1
fi

# 运行主函数
main "$@"