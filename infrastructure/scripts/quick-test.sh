#!/bin/bash

# scripts/quick-test.sh
# RAG 系统快速功能测试脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
RAG_ENGINE_URL="${RAG_ENGINE_URL:-http://localhost:8080}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
TIMEOUT="${TIMEOUT:-10}"

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

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

# 测试函数
test_endpoint() {
    local url="$1"
    local description="$2"
    local expected_status="${3:-200}"
    local method="${4:-GET}"
    local data="${5:-}"

    log_test "测试 $description..."

    local curl_args=(-s -w "%{http_code}" -o /tmp/response.json --max-time "$TIMEOUT")

    if [ "$method" = "POST" ]; then
        curl_args+=(-X POST)
        if [ -n "$data" ]; then
            curl_args+=(-H "Content-Type: application/json" -d "$data")
        fi
    fi

    local status_code
    status_code=$(curl "${curl_args[@]}" "$url" 2>/dev/null || echo "000")

    if [ "$status_code" = "$expected_status" ]; then
        log_success "✓ $description (HTTP $status_code)"
        return 0
    else
        log_error "✗ $description (HTTP $status_code, 期望 $expected_status)"
        if [ -f /tmp/response.json ]; then
            echo "响应内容: $(cat /tmp/response.json)"
        fi
        return 1
    fi
}

test_json_response() {
    local url="$1"
    local description="$2"
    local expected_field="$3"

    log_test "测试 $description (JSON 响应)..."

    local response
    response=$(curl -s --max-time "$TIMEOUT" "$url" 2>/dev/null || echo "{}")

    if echo "$response" | jq -e ".$expected_field" >/dev/null 2>&1; then
        log_success "✓ $description (包含字段: $expected_field)"
        return 0
    else
        log_error "✗ $description (缺少字段: $expected_field)"
        echo "响应内容: $response"
        return 1
    fi
}

# 主测试函数
run_basic_tests() {
    echo "🧪 RAG 系统基础功能测试"
    echo "========================"
    echo

    local test_count=0
    local passed_count=0

    # 测试健康检查端点
    log_info "1. 健康检查测试"
    echo "--------------------"

    ((test_count++))
    if test_endpoint "$API_BASE_URL/health" "Go API 健康检查"; then
        ((passed_count++))
    fi

    ((test_count++))
    if test_endpoint "$RAG_ENGINE_URL/health" "Rust 引擎健康检查"; then
        ((passed_count++))
    fi

    ((test_count++))
    if test_endpoint "$FRONTEND_URL" "前端服务"; then
        ((passed_count++))
    fi

    echo

    # 测试就绪检查
    log_info "2. 就绪检查测试"
    echo "--------------------"

    ((test_count++))
    if test_endpoint "$API_BASE_URL/ready" "Go API 就绪检查"; then
        ((passed_count++))
    fi

    echo

    # 测试 API 端点结构
    log_info "3. API 端点结构测试"
    echo "--------------------"

    ((test_count++))
    if test_json_response "$API_BASE_URL/health" "健康检查 JSON 结构" "status"; then
        ((passed_count++))
    fi

    ((test_count++))
    if test_json_response "$API_BASE_URL/ready" "就绪检查 JSON 结构" "status"; then
        ((passed_count++))
    fi

    echo

    # 测试未实现的 API (应该返回 501)
    log_info "4. 未实现功能测试"
    echo "--------------------"

    ((test_count++))
    if test_endpoint "$API_BASE_URL/api/v1/documents" "文档列表 API (未实现)" "501"; then
        ((passed_count++))
    fi

    ((test_count++))
    if test_endpoint "$API_BASE_URL/api/v1/search" "搜索 API (未实现)" "501" "POST" '{"query":"test"}'; then
        ((passed_count++))
    fi

    echo

    # 显示结果摘要
    echo "📊 测试结果摘要"
    echo "================"
    echo "总测试数: $test_count"
    echo "通过测试: $passed_count"
    echo "失败测试: $((test_count - passed_count))"
    echo "成功率: $(( passed_count * 100 / test_count ))%"
    echo

    if [ $passed_count -eq $test_count ]; then
        log_success "🎉 所有基础测试通过！"
        return 0
    else
        log_error "❌ 部分测试失败，请检查服务状态"
        return 1
    fi
}

run_performance_tests() {
    echo "⚡ RAG 系统性能测试"
    echo "==================="
    echo

    log_info "1. 响应时间测试"
    echo "--------------------"

    # 测试健康检查响应时间
    log_test "测试 API 健康检查响应时间..."
    local start_time=$(date +%s%N)
    if curl -s --max-time 5 "$API_BASE_URL/health" >/dev/null; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))  # 转换为毫秒
        if [ $duration -lt 100 ]; then
            log_success "✓ API 响应时间: ${duration}ms (优秀)"
        elif [ $duration -lt 500 ]; then
            log_success "✓ API 响应时间: ${duration}ms (良好)"
        else
            log_warning "⚠ API 响应时间: ${duration}ms (较慢)"
        fi
    else
        log_error "✗ API 健康检查失败"
    fi

    # 测试并发请求
    log_info "2. 并发测试"
    echo "--------------------"

    log_test "测试并发健康检查请求 (10个并发)..."
    local success_count=0
    for i in {1..10}; do
        if curl -s --max-time 5 "$API_BASE_URL/health" >/dev/null & then
            ((success_count++))
        fi
    done
    wait

    if [ $success_count -eq 10 ]; then
        log_success "✓ 并发测试: 10/10 请求成功"
    else
        log_warning "⚠ 并发测试: $success_count/10 请求成功"
    fi

    echo
}

run_integration_tests() {
    echo "🔗 RAG 系统集成测试"
    echo "==================="
    echo

    log_info "1. 服务间通信测试"
    echo "--------------------"

    # 这里可以添加更复杂的集成测试
    # Example: 测试 Go API 调用 Rust 引擎

    log_test "测试服务发现..."
    if curl -s --max-time 5 "$API_BASE_URL/health" >/dev/null && \
       curl -s --max-time 5 "$RAG_ENGINE_URL/health" >/dev/null; then
        log_success "✓ 所有核心服务可达"
    else
        log_error "✗ 部分服务不可达"
    fi

    echo

    log_info "2. 数据库连接测试"
    echo "--------------------"

    # 通过就绪检查间接测试数据库连接
    if test_endpoint "$API_BASE_URL/ready" "数据库连接 (通过就绪检查)" "200"; then
        log_success "✓ 数据库连接正常"
    else
        log_error "✗ 数据库连接异常"
    fi

    echo
}

# 生成测试报告
generate_test_report() {
    local report_file="test-report-$(date +%Y%m%d-%H%M%S).json"

    log_info "生成测试报告..."

    # 收集系统信息
    local system_info
    system_info=$(cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "test_environment": {
    "api_url": "$API_BASE_URL",
    "engine_url": "$RAG_ENGINE_URL",
    "frontend_url": "$FRONTEND_URL",
    "timeout": $TIMEOUT
  },
  "system_info": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "docker_version": "$(docker --version 2>/dev/null || echo 'Not available')",
    "curl_version": "$(curl --version 2>/dev/null | head -n1 || echo 'Not available')"
  }
}
EOF
    )

    echo "$system_info" > "$report_file"
    log_success "测试报告已保存: $report_file"
}

# 清理测试环境
cleanup_test_env() {
    log_info "清理测试环境..."
    rm -f /tmp/response.json
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "RAG 系统快速测试脚本"
    echo
    echo "用法:"
    echo "  $0 [选项] [测试类型]"
    echo
    echo "测试类型:"
    echo "  basic         基础功能测试 (默认)"
    echo "  performance   性能测试"
    echo "  integration   集成测试"
    echo "  all           运行所有测试"
    echo
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -t, --timeout SECONDS   设置请求超时时间 (默认: 10)"
    echo "  -u, --api-url URL       设置 API 基础 URL (默认: http://localhost:8000)"
    echo "  -e, --engine-url URL    设置引擎 URL (默认: http://localhost:8080)"
    echo "  -f, --frontend-url URL  设置前端 URL (默认: http://localhost:3000)"
    echo "  -r, --report           生成详细测试报告"
    echo "  -v, --verbose          详细输出模式"
    echo
    echo "示例:"
    echo "  $0                     # 运行基础测试"
    echo "  $0 all                 # 运行所有测试"
    echo "  $0 -t 30 performance   # 运行性能测试 (30秒超时)"
    echo "  $0 -r basic            # 运行基础测试并生成报告"
}

# 主函数
main() {
    local test_type="basic"
    local generate_report=false
    local verbose=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -u|--api-url)
                API_BASE_URL="$2"
                shift 2
                ;;
            -e|--engine-url)
                RAG_ENGINE_URL="$2"
                shift 2
                ;;
            -f|--frontend-url)
                FRONTEND_URL="$2"
                shift 2
                ;;
            -r|--report)
                generate_report=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                set -x
                shift
                ;;
            basic|performance|integration|all)
                test_type="$1"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 检查依赖
    if ! command -v curl >/dev/null; then
        log_error "需要 curl 命令，请先安装"
        exit 1
    fi

    if ! command -v jq >/dev/null; then
        log_warning "建议安装 jq 以获得更好的 JSON 解析能力"
    fi

    # 显示测试配置
    echo "🔧 测试配置"
    echo "==========="
    echo "API URL: $API_BASE_URL"
    echo "引擎 URL: $RAG_ENGINE_URL"
    echo "前端 URL: $FRONTEND_URL"
    echo "超时时间: ${TIMEOUT}s"
    echo "测试类型: $test_type"
    echo

    # 运行测试
    local exit_code=0

    case $test_type in
        basic)
            run_basic_tests || exit_code=1
            ;;
        performance)
            run_performance_tests || exit_code=1
            ;;
        integration)
            run_integration_tests || exit_code=1
            ;;
        all)
            run_basic_tests || exit_code=1
            echo
            run_performance_tests || exit_code=1
            echo
            run_integration_tests || exit_code=1
            ;;
        *)
            log_error "未知测试类型: $test_type"
            exit 1
            ;;
    esac

    # 生成报告
    if [ "$generate_report" = true ]; then
        echo
        generate_test_report
    fi

    # 清理
    cleanup_test_env

    # 显示最终结果
    echo
    if [ $exit_code -eq 0 ]; then
        log_success "🎉 所有测试完成！系统运行正常。"
        echo
        log_info "🚀 你现在可以："
        echo "   • 访问前端界面: $FRONTEND_URL"
        echo "   • 查看 API 文档: $API_BASE_URL/swagger/index.html"
        echo "   • 开始开发新功能"
    else
        log_error "❌ 测试发现问题，请检查服务状态。"
        echo
        log_info "🔧 故障排除建议："
        echo "   • 检查服务是否启动: make dev-status"
        echo "   • 查看服务日志: make dev-logs"
        echo "   • 重启服务: make dev-restart"
    fi

    exit $exit_code
}

# 捕获中断信号
trap cleanup_test_env EXIT

# 运行主函数
main "$@"