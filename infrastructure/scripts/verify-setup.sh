#!/bin/bash

# scripts/verify-setup.sh
# RAG 系统设置验证脚本

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

# 验证函数
verify_file() {
    local file_path="$1"
    local description="$2"

    if [ -f "$file_path" ]; then
        log_success "✓ $description: $file_path"
        return 0
    else
        log_error "✗ $description: $file_path (文件不存在)"
        return 1
    fi
}

verify_directory() {
    local dir_path="$1"
    local description="$2"

    if [ -d "$dir_path" ]; then
        log_success "✓ $description: $dir_path"
        return 0
    else
        log_error "✗ $description: $dir_path (目录不存在)"
        return 1
    fi
}

verify_command() {
    local command="$1"
    local description="$2"

    if command -v "$command" >/dev/null 2>&1; then
        local version=$(eval "$command --version 2>/dev/null | head -n1" || echo "unknown")
        log_success "✓ $description: $command ($version)"
        return 0
    else
        log_error "✗ $description: $command (命令不存在)"
        return 1
    fi
}

verify_service() {
    local url="$1"
    local description="$2"
    local timeout="${3:-5}"

    if curl -s --max-time "$timeout" "$url" >/dev/null 2>&1; then
        log_success "✓ $description: $url"
        return 0
    else
        log_warning "⚠ $description: $url (服务不可用或未启动)"
        return 1
    fi
}

# 主验证函数
main() {
    echo "🔍 RAG 系统设置验证"
    echo "======================"
    echo

    local error_count=0

    log_info "检查项目结构..."

    # 验证主要目录
    verify_directory "rag-engine" "Rust RAG 引擎目录" || ((error_count++))
    verify_directory "rag-api" "Go API 网关目录" || ((error_count++))
    verify_directory "rag-frontend" "React 前端目录" || ((error_count++))
    verify_directory "infrastructure" "基础设施配置目录" || ((error_count++))
    verify_directory "scripts" "脚本目录" || ((error_count++))

    echo

    log_info "检查配置文件..."

    # 验证配置文件
    verify_file "rag-engine/Cargo.toml" "Rust 项目配置" || ((error_count++))
    verify_file "rag-engine/config/local.toml" "Rust 本地配置" || ((error_count++))
    verify_file "rag-api/go.mod" "Go 模块配置" || ((error_count++))
    verify_file "rag-api/configs/local.yaml" "Go 本地配置" || ((error_count++))
    verify_file "rag-frontend/package.json" "前端项目配置" || ((error_count++))
    verify_file ".env.example" "环境变量模板" || ((error_count++))

    echo

    log_info "检查核心源代码文件..."

    # Rust 核心文件
    verify_file "rag-engine/src/lib.rs" "Rust 库入口文件" || ((error_count++))
    verify_file "rag-engine/src/main.rs" "Rust 主程序文件" || ((error_count++))
    verify_file "rag-engine/src/config/mod.rs" "Rust 配置模块" || ((error_count++))
    verify_file "rag-engine/src/error/mod.rs" "Rust 错误处理模块" || ((error_count++))
    verify_file "rag-engine/src/types/mod.rs" "Rust 类型定义模块" || ((error_count++))
    verify_file "rag-engine/src/services/mod.rs" "Rust 服务模块" || ((error_count++))
    verify_file "rag-engine/src/services/rag_service.rs" "Rust RAG 核心服务" || ((error_count++))
    verify_file "rag-engine/src/cache/mod.rs" "Rust 缓存模块" || ((error_count++))
    verify_file "rag-engine/src/network/mod.rs" "Rust 网络模块" || ((error_count++))
    verify_file "rag-engine/src/network/http.rs" "Rust HTTP 服务器" || ((error_count++))
    verify_file "rag-engine/src/network/grpc.rs" "Rust gRPC 服务器" || ((error_count++))
    verify_file "rag-engine/src/network/websocket.rs" "Rust WebSocket 服务器" || ((error_count++))

    # Go 核心文件
    verify_file "rag-api/cmd/server/main.go" "Go 服务器启动文件" || ((error_count++))
    verify_file "rag-api/internal/config/config.go" "Go 配置模块" || ((error_count++))
    verify_file "rag-api/pkg/logger/logger.go" "Go 日志模块" || ((error_count++))
    verify_file "rag-api/internal/gateway/server.go" "Go 网关服务器" || ((error_count++))
    verify_file "rag-api/internal/handlers/health.go" "Go 健康检查处理器" || ((error_count++))
    verify_file "rag-api/internal/handlers/handlers.go" "Go 通用处理器" || ((error_count++))

    echo

    log_info "检查 Docker 配置..."

    # Docker 配置文件
    verify_file "infrastructure/docker/docker-compose/development.yml" "开发环境 Docker Compose" || ((error_count++))
    verify_file "infrastructure/docker/rag-engine/Dockerfile.dev" "Rust 引擎开发 Dockerfile" || ((error_count++))
    verify_file "infrastructure/docker/rag-api/Dockerfile.dev" "Go API 开发 Dockerfile" || ((error_count++))
    verify_file "infrastructure/docker/rag-frontend/Dockerfile.dev" "前端开发 Dockerfile" || ((error_count++))
    verify_file "infrastructure/docker/rag-frontend/nginx.conf" "Nginx 配置文件" || ((error_count++))

    echo

    log_info "检查管理脚本..."

    # 管理脚本
    verify_file "scripts/dev-start.sh" "开发环境启动脚本" || ((error_count++))
    verify_file "Makefile" "项目管理 Makefile" || ((error_count++))
    verify_file "README.md" "项目说明文档" || ((error_count++))

    echo

    log_info "检查开发工具..."

    # 开发工具检查
    verify_command "docker" "Docker 容器引擎"
    verify_command "docker-compose" "Docker Compose 编排工具"
    verify_command "rustc" "Rust 编译器"
    verify_command "cargo" "Rust 包管理器"
    verify_command "go" "Go 编译器"
    verify_command "node" "Node.js 运行时"
    verify_command "npm" "Node.js 包管理器"
    verify_command "make" "Make 构建工具"
    verify_command "curl" "HTTP 客户端工具"
    verify_command "git" "Git 版本控制"

    echo

    log_info "检查环境变量配置..."

    # 检查 .env 文件
    if [ -f ".env" ]; then
        log_success "✓ 环境变量文件: .env"

        # 检查关键环境变量
        if grep -q "DATABASE_URL" .env; then
            log_success "  ✓ 数据库配置已设置"
        else
            log_warning "  ⚠ 数据库配置未设置"
        fi

        if grep -q "REDIS_URL" .env; then
            log_success "  ✓ Redis 配置已设置"
        else
            log_warning "  ⚠ Redis 配置未设置"
        fi

        if grep -q "OPENAI_API_KEY" .env; then
            log_success "  ✓ OpenAI API 密钥已设置"
        else
            log_warning "  ⚠ OpenAI API 密钥未设置"
        fi
    else
        log_warning "⚠ 环境变量文件不存在，请复制 .env.example 为 .env"
    fi

    echo

    log_info "检查 Docker 服务状态..."

    # 检查 Docker 是否运行
    if docker info >/dev/null 2>&1; then
        log_success "✓ Docker 服务正在运行"

        # 检查 Docker Compose 服务
        if docker-compose -f infrastructure/docker/docker-compose/development.yml ps >/dev/null 2>&1; then
            log_info "Docker Compose 服务状态："
            docker-compose -f infrastructure/docker/docker-compose/development.yml ps
        else
            log_warning "⚠ Docker Compose 服务未启动"
        fi
    else
        log_warning "⚠ Docker 服务未运行"
    fi

    echo

    log_info "检查服务端点..."

    # 检查服务端点（如果服务正在运行）
    verify_service "http://localhost:8000/health" "Go API 健康检查"
    verify_service "http://localhost:8080/health" "Rust 引擎健康检查"
    verify_service "http://localhost:3000" "React 前端服务"
    verify_service "http://localhost:6333" "Qdrant 向量数据库"
    verify_service "http://localhost:5432" "PostgreSQL 数据库" 2
    verify_service "http://localhost:6379" "Redis 缓存" 2
    verify_service "http://localhost:9000" "MinIO 对象存储"

    echo

    log_info "生成验证报告..."

    # 创建验证报告
    report_file="verification-report-$(date +%Y%m%d-%H%M%S).txt"
    {
        echo "RAG 系统设置验证报告"
        echo "======================"
        echo "生成时间: $(date)"
        echo
        echo "验证结果摘要："
        if [ $error_count -eq 0 ]; then
            echo "✅ 所有必需文件和配置都已正确设置"
        else
            echo "❌ 发现 $error_count 个问题需要解决"
        fi
        echo
        echo "详细验证日志："
        echo "（请查看控制台输出）"
    } > "$report_file"

    echo

    # 最终结果
    if [ $error_count -eq 0 ]; then
        log_success "🎉 验证完成！所有核心文件和配置都已正确设置。"
        log_info "📋 验证报告已保存到: $report_file"
        echo
        log_info "🚀 你现在可以启动开发环境了："
        echo "   make dev              # 启动所有服务"
        echo "   make dev-status       # 检查服务状态"
        echo "   make dev-logs         # 查看服务日志"
        echo
        log_info "📚 更多命令请查看："
        echo "   make help             # 显示所有可用命令"
    else
        log_error "❌ 验证发现 $error_count 个问题。"
        log_info "📋 验证报告已保存到: $report_file"
        echo
        log_info "🔧 请解决以下问题后重新运行验证："
        echo "   1. 检查缺失的文件和目录"
        echo "   2. 安装缺失的开发工具"
        echo "   3. 配置环境变量文件 (.env)"
        echo "   4. 重新运行: ./scripts/verify-setup.sh"
        echo
        exit 1
    fi
}

# 帮助信息
show_help() {
    echo "RAG 系统设置验证脚本"
    echo
    echo "用法:"
    echo "  $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -v, --verbose  详细输出模式"
    echo "  --services     仅检查服务端点"
    echo "  --files        仅检查文件和目录"
    echo "  --tools        仅检查开发工具"
    echo
    echo "示例:"
    echo "  $0              # 完整验证"
    echo "  $0 --services   # 仅检查服务"
    echo "  $0 --files      # 仅检查文件"
}

# 仅检查服务
check_services_only() {
    log_info "检查服务端点..."
    verify_service "http://localhost:8000/health" "Go API 健康检查"
    verify_service "http://localhost:8080/health" "Rust 引擎健康检查"
    verify_service "http://localhost:3000" "React 前端服务"
    verify_service "http://localhost:6333" "Qdrant 向量数据库"
    verify_service "http://localhost:5432" "PostgreSQL 数据库" 2
    verify_service "http://localhost:6379" "Redis 缓存" 2
    verify_service "http://localhost:9000" "MinIO 对象存储"
}

# 仅检查文件
check_files_only() {
    log_info "检查核心文件和目录..."
    local error_count=0

    # 验证主要目录
    verify_directory "rag-engine" "Rust RAG 引擎目录" || ((error_count++))
    verify_directory "rag-api" "Go API 网关目录" || ((error_count++))
    verify_directory "rag-frontend" "React 前端目录" || ((error_count++))
    verify_directory "infrastructure" "基础设施配置目录" || ((error_count++))

    # 验证配置文件
    verify_file "rag-engine/Cargo.toml" "Rust 项目配置" || ((error_count++))
    verify_file "rag-api/go.mod" "Go 模块配置" || ((error_count++))
    verify_file "rag-frontend/package.json" "前端项目配置" || ((error_count++))
    verify_file ".env.example" "环境变量模板" || ((error_count++))

    if [ $error_count -eq 0 ]; then
        log_success "✅ 所有核心文件和目录都存在"
    else
        log_error "❌ 发现 $error_count 个文件/目录问题"
    fi
}

# 仅检查工具
check_tools_only() {
    log_info "检查开发工具..."
    verify_command "docker" "Docker 容器引擎"
    verify_command "docker-compose" "Docker Compose 编排工具"
    verify_command "rustc" "Rust 编译器"
    verify_command "cargo" "Rust 包管理器"
    verify_command "go" "Go 编译器"
    verify_command "node" "Node.js 运行时"
    verify_command "npm" "Node.js 包管理器"
    verify_command "make" "Make 构建工具"
    verify_command "curl" "HTTP 客户端工具"
    verify_command "git" "Git 版本控制"
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --services)
        check_services_only
        exit 0
        ;;
    --files)
        check_files_only
        exit 0
        ;;
    --tools)
        check_tools_only
        exit 0
        ;;
    -v|--verbose)
        set -x
        main
        ;;
    "")
        main
        ;;
    *)
        log_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac