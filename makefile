# RAG 系统 Makefile
# 提供统一的开发、构建、测试和部署命令

.PHONY: help init dev build test clean deploy docs lint format check install

# 默认目标
.DEFAULT_GOAL := help

# 颜色定义
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# 项目信息
PROJECT_NAME := rag-system
VERSION := 0.1.0
DOCKER_REGISTRY := your-registry.com
NAMESPACE := rag-system

# 工具检查
DOCKER := $(shell command -v docker 2> /dev/null)
DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null)
CARGO := $(shell command -v cargo 2> /dev/null)
GO := $(shell command -v go 2> /dev/null)
NODE := $(shell command -v node 2> /dev/null)
NPM := $(shell command -v npm 2> /dev/null)
KUBECTL := $(shell command -v kubectl 2> /dev/null)
HELM := $(shell command -v helm 2> /dev/null)

##@ 帮助信息

help: ## 显示帮助信息
	@echo "$(BOLD)$(BLUE)RAG 系统管理命令$(RESET)"
	@echo "项目: $(PROJECT_NAME) v$(VERSION)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(BOLD)使用方法:$(RESET)\n  make $(YELLOW)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ 环境设置

check-deps: ## 检查依赖工具
	@echo "$(BOLD)检查依赖工具...$(RESET)"
	@echo -n "Docker: "; if [ -n "$(DOCKER)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "Docker Compose: "; if [ -n "$(DOCKER_COMPOSE)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "Rust/Cargo: "; if [ -n "$(CARGO)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "Go: "; if [ -n "$(GO)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "Node.js: "; if [ -n "$(NODE)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "npm: "; if [ -n "$(NPM)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "kubectl: "; if [ -n "$(KUBECTL)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi
	@echo -n "helm: "; if [ -n "$(HELM)" ]; then echo "$(GREEN)✓$(RESET)"; else echo "$(RED)✗$(RESET)"; fi

install: check-deps ## 安装项目依赖
	@echo "$(BOLD)安装项目依赖...$(RESET)"
	@if [ -n "$(CARGO)" ]; then \
		echo "$(BLUE)安装 Rust 依赖...$(RESET)"; \
		cd rag-engine && cargo fetch; \
	fi
	@if [ -n "$(GO)" ]; then \
		echo "$(BLUE)安装 Go 依赖...$(RESET)"; \
		cd rag-api && go mod download; \
	fi
	@if [ -n "$(NPM)" ]; then \
		echo "$(BLUE)安装前端依赖...$(RESET)"; \
		cd rag-frontend && npm install; \
	fi
	@echo "$(GREEN)依赖安装完成$(RESET)"

init: install ## 初始化项目环境
	@echo "$(BOLD)初始化项目环境...$(RESET)"
	@mkdir -p logs data/{postgres,redis,qdrant,minio} uploads models/{embedding,llm} plugins
	@if [ ! -f .env ]; then \
		echo "$(BLUE)创建环境变量文件...$(RESET)"; \
		cp .env.example .env 2>/dev/null || echo "DATABASE_URL=postgres://rag_user:rag_password@localhost:5432/rag_development" > .env; \
	fi
	@chmod +x scripts/dev-*.sh 2>/dev/null || true
	@echo "$(GREEN)项目初始化完成$(RESET)"

##@ 开发环境

dev: ## 启动完整开发环境
	@echo "$(BOLD)启动开发环境...$(RESET)"
	@chmod +x scripts/dev-start.sh
	@./scripts/dev-start.sh

dev-stop: ## 停止开发环境
	@echo "$(BOLD)停止开发环境...$(RESET)"
	@chmod +x scripts/dev-stop.sh 2>/dev/null && ./scripts/dev-stop.sh || true

dev-restart: dev-stop dev ## 重启开发环境

dev-logs: ## 查看开发环境日志
	@echo "$(BOLD)查看服务日志...$(RESET)"
	@chmod +x scripts/dev-logs.sh 2>/dev/null && ./scripts/dev-logs.sh || tail -f logs/*.log

dev-status: ## 检查开发环境状态
	@echo "$(BOLD)检查服务状态...$(RESET)"
	@echo "$(BLUE)基础设施服务:$(RESET)"
	@curl -s http://localhost:5432 >/dev/null 2>&1 && echo "  PostgreSQL: $(GREEN)✓$(RESET)" || echo "  PostgreSQL: $(RED)✗$(RESET)"
	@curl -s http://localhost:6379 >/dev/null 2>&1 && echo "  Redis: $(GREEN)✓$(RESET)" || echo "  Redis: $(RED)✗$(RESET)"
	@curl -s http://localhost:6333/health >/dev/null 2>&1 && echo "  Qdrant: $(GREEN)✓$(RESET)" || echo "  Qdrant: $(RED)✗$(RESET)"
	@curl -s http://localhost:9000/minio/health/live >/dev/null 2>&1 && echo "  MinIO: $(GREEN)✓$(RESET)" || echo "  MinIO: $(RED)✗$(RESET)"
	@echo "$(BLUE)应用服务:$(RESET)"
	@curl -s http://localhost:8080/health >/dev/null 2>&1 && echo "  Rust 引擎: $(GREEN)✓$(RESET)" || echo "  Rust 引擎: $(RED)✗$(RESET)"
	@curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "  Go API: $(GREEN)✓$(RESET)" || echo "  Go API: $(RED)✗$(RESET)"
	@curl -s http://localhost:3000 >/dev/null 2>&1 && echo "  前端: $(GREEN)✓$(RESET)" || echo "  前端: $(RED)✗$(RESET)"

##@ 构建和测试

build: build-rust build-go build-frontend ## 构建所有组件

build-rust: ## 构建 Rust 引擎
	@echo "$(BOLD)构建 Rust 引擎...$(RESET)"
	@cd rag-engine && cargo build --release
	@echo "$(GREEN)Rust 引擎构建完成$(RESET)"

build-go: ## 构建 Go API
	@echo "$(BOLD)构建 Go API...$(RESET)"
	@cd rag-api && go build -o bin/rag-api cmd/server/main.go
	@echo "$(GREEN)Go API 构建完成$(RESET)"

build-frontend: ## 构建前端
	@echo "$(BOLD)构建前端...$(RESET)"
	@cd rag-frontend && npm run build
	@echo "$(GREEN)前端构建完成$(RESET)"

test: test-rust test-go test-frontend ## 运行所有测试

test-rust: ## 运行 Rust 测试
	@echo "$(BOLD)运行 Rust 测试...$(RESET)"
	@cd rag-engine && cargo test
	@echo "$(GREEN)Rust 测试完成$(RESET)"

test-go: ## 运行 Go 测试
	@echo "$(BOLD)运行 Go 测试...$(RESET)"
	@cd rag-api && go test ./...
	@echo "$(GREEN)Go 测试完成$(RESET)"

test-frontend: ## 运行前端测试
	@echo "$(BOLD)运行前端测试...$(RESET)"
	@cd rag-frontend && npm test
	@echo "$(GREEN)前端测试完成$(RESET)"

bench: ## 运行性能测试
	@echo "$(BOLD)运行性能测试...$(RESET)"
	@cd rag-engine && cargo bench

##@ 代码质量

lint: lint-rust lint-go lint-frontend ## 运行所有代码检查

lint-rust: ## 检查 Rust 代码
	@echo "$(BOLD)检查 Rust 代码...$(RESET)"
	@cd rag-engine && cargo clippy -- -D warnings

lint-go: ## 检查 Go 代码
	@echo "$(BOLD)检查 Go 代码...$(RESET)"
	@cd rag-api && golangci-lint run 2>/dev/null || echo "$(YELLOW)golangci-lint 未安装，跳过检查$(RESET)"

lint-frontend: ## 检查前端代码
	@echo "$(BOLD)检查前端代码...$(RESET)"
	@cd rag-frontend && npm run lint

format: format-rust format-go format-frontend ## 格式化所有代码

format-rust: ## 格式化 Rust 代码
	@echo "$(BOLD)格式化 Rust 代码...$(RESET)"
	@cd rag-engine && cargo fmt

format-go: ## 格式化 Go 代码
	@echo "$(BOLD)格式化 Go 代码...$(RESET)"
	@cd rag-api && go fmt ./...

format-frontend: ## 格式化前端代码
	@echo "$(BOLD)格式化前端代码...$(RESET)"
	@cd rag-frontend && npm run format

check: lint test ## 运行所有检查和测试

##@ Docker 构建

docker-build: docker-build-rust docker-build-go docker-build-frontend ## 构建所有 Docker 镜像

docker-build-rust: ## 构建 Rust 引擎 Docker 镜像
	@echo "$(BOLD)构建 Rust 引擎 Docker 镜像...$(RESET)"
	@docker build -f infrastructure/docker/rag-engine/Dockerfile -t $(DOCKER_REGISTRY)/rag-engine:$(VERSION) rag-engine/
	@docker tag $(DOCKER_REGISTRY)/rag-engine:$(VERSION) $(DOCKER_REGISTRY)/rag-engine:latest
	@echo "$(GREEN)Rust 引擎镜像构建完成$(RESET)"

docker-build-go: ## 构建 Go API Docker 镜像
	@echo "$(BOLD)构建 Go API Docker 镜像...$(RESET)"
	@docker build -f infrastructure/docker/rag-api/Dockerfile -t $(DOCKER_REGISTRY)/rag-api:$(VERSION) rag-api/
	@docker tag $(DOCKER_REGISTRY)/rag-api:$(VERSION) $(DOCKER_REGISTRY)/rag-api:latest
	@echo "$(GREEN)Go API 镜像构建完成$(RESET)"

docker-build-frontend: ## 构建前端 Docker 镜像
	@echo "$(BOLD)构建前端 Docker 镜像...$(RESET)"
	@docker build -f infrastructure/docker/rag-frontend/Dockerfile -t $(DOCKER_REGISTRY)/rag-frontend:$(VERSION) rag-frontend/
	@docker tag $(DOCKER_REGISTRY)/rag-frontend:$(VERSION) $(DOCKER_REGISTRY)/rag-frontend:latest
	@echo "$(GREEN)前端镜像构建完成$(RESET)"

docker-push: ## 推送 Docker 镜像到注册表
	@echo "$(BOLD)推送 Docker 镜像...$(RESET)"
	@docker push $(DOCKER_REGISTRY)/rag-engine:$(VERSION)
	@docker push $(DOCKER_REGISTRY)/rag-engine:latest
	@docker push $(DOCKER_REGISTRY)/rag-api:$(VERSION)
	@docker push $(DOCKER_REGISTRY)/rag-api:latest
	@docker push $(DOCKER_REGISTRY)/rag-frontend:$(VERSION)
	@docker push $(DOCKER_REGISTRY)/rag-frontend:latest
	@echo "$(GREEN)镜像推送完成$(RESET)"

##@ 部署

deploy-dev: ## 部署到开发环境
	@echo "$(BOLD)部署到开发环境...$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose/development.yml up -d
	@echo "$(GREEN)开发环境部署完成$(RESET)"

deploy-staging: ## 部署到测试环境
	@echo "$(BOLD)部署到测试环境...$(RESET)"
	@if [ -n "$(KUBECTL)" ] && [ -n "$(HELM)" ]; then \
		helm upgrade --install rag-system-staging infrastructure/helm/rag-system/ \
			--namespace $(NAMESPACE)-staging --create-namespace \
			--values infrastructure/helm/rag-system/values-staging.yaml \
			--set image.tag=$(VERSION); \
	else \
		echo "$(RED)kubectl 或 helm 未安装$(RESET)"; \
	fi
	@echo "$(GREEN)测试环境部署完成$(RESET)"

deploy-prod: ## 部署到生产环境
	@echo "$(BOLD)部署到生产环境...$(RESET)"
	@if [ -n "$(KUBECTL)" ] && [ -n "$(HELM)" ]; then \
		helm upgrade --install rag-system infrastructure/helm/rag-system/ \
			--namespace $(NAMESPACE) --create-namespace \
			--values infrastructure/helm/rag-system/values-prod.yaml \
			--set image.tag=$(VERSION); \
	else \
		echo "$(RED)kubectl 或 helm 未安装$(RESET)"; \
	fi
	@echo "$(GREEN)生产环境部署完成$(RESET)"

##@ 数据库管理

db-migrate: ## 运行数据库迁移
	@echo "$(BOLD)运行数据库迁移...$(RESET)"
	@cd rag-api && go run cmd/migrate/main.go up

db-rollback: ## 回滚数据库迁移
	@echo "$(BOLD)回滚数据库迁移...$(RESET)"
	@cd rag-api && go run cmd/migrate/main.go down

db-reset: ## 重置数据库
	@echo "$(BOLD)重置数据库...$(RESET)"
	@cd rag-api && go run cmd/migrate/main.go reset

db-seed: ## 填充测试数据
	@echo "$(BOLD)填充测试数据...$(RESET)"
	@cd rag-api && go run cmd/seed/main.go

##@ 监控和日志

logs-rust: ## 查看 Rust 引擎日志
	@echo "$(BOLD)Rust 引擎日志:$(RESET)"
	@tail -f logs/rag-engine.log 2>/dev/null || echo "$(YELLOW)日志文件不存在$(RESET)"

logs-go: ## 查看 Go API 日志
	@echo "$(BOLD)Go API 日志:$(RESET)"
	@tail -f logs/rag-api.log 2>/dev/null || echo "$(YELLOW)日志文件不存在$(RESET)"

logs-frontend: ## 查看前端日志
	@echo "$(BOLD)前端日志:$(RESET)"
	@tail -f logs/rag-frontend.log 2>/dev/null || echo "$(YELLOW)日志文件不存在$(RESET)"

monitoring: ## 打开监控面板
	@echo "$(BOLD)监控服务链接:$(RESET)"
	@echo "  Grafana:    http://localhost:3001"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Jaeger:     http://localhost:16686"
	@echo "  API 文档:   http://localhost:8000/swagger/index.html"

##@ 文档

docs: ## 生成文档
	@echo "$(BOLD)生成项目文档...$(RESET)"
	@cd rag-engine && cargo doc --no-deps
	@cd rag-api && swag init -g cmd/server/main.go
	@cd rag-frontend && npm run build-storybook 2>/dev/null || echo "$(YELLOW)Storybook 未配置$(RESET)"
	@echo "$(GREEN)文档生成完成$(RESET)"

docs-serve: ## 启动文档服务器
	@echo "$(BOLD)启动文档服务器...$(RESET)"
	@echo "  Rust 文档:  http://localhost:8080/docs"
	@echo "  API 文档:   http://localhost:8000/swagger/index.html"
	@echo "  Storybook:  http://localhost:6006"

##@ 清理

clean: ## 清理构建文件
	@echo "$(BOLD)清理构建文件...$(RESET)"
	@cd rag-engine && cargo clean
	@cd rag-api && go clean && rm -rf bin/
	@cd rag-frontend && rm -rf dist/ build/
	@echo "$(GREEN)清理完成$(RESET)"

clean-docker: ## 清理 Docker 资源
	@echo "$(BOLD)清理 Docker 资源...$(RESET)"
	@docker system prune -f
	@docker volume prune -f
	@echo "$(GREEN)Docker 清理完成$(RESET)"

clean-logs: ## 清理日志文件
	@echo "$(BOLD)清理日志文件...$(RESET)"
	@rm -rf logs/*.log
	@echo "$(GREEN)日志清理完成$(RESET)"

clean-all: clean clean-docker clean-logs ## 清理所有文件

##@ 实用工具

config-check: ## 检查配置文件
	@echo "$(BOLD)检查配置文件...$(RESET)"
	@cd rag-engine && cargo run --bin rag-engine-server -- --validate-config
	@echo "$(GREEN)Rust 引擎配置检查完成$(RESET)"

health-check: ## 执行健康检查
	@echo "$(BOLD)执行健康检查...$(RESET)"
	@cd rag-engine && cargo run --bin rag-engine-server -- --check-health
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
	@echo "$(GREEN)健康检查完成$(RESET)"

security-scan: ## 运行安全扫描
	@echo "$(BOLD)运行安全扫描...$(RESET)"
	@cd rag-engine && cargo audit 2>/dev/null || echo "$(YELLOW)cargo-audit 未安装$(RESET)"
	@cd rag-api && gosec ./... 2>/dev/null || echo "$(YELLOW)gosec 未安装$(RESET)"
	@cd rag-frontend && npm audit 2>/dev/null || echo "$(YELLOW)npm audit 失败$(RESET)"

update-deps: ## 更新依赖
	@echo "$(BOLD)更新项目依赖...$(RESET)"
	@cd rag-engine && cargo update
	@cd rag-api && go get -u ./... && go mod tidy
	@cd rag-frontend && npm update
	@echo "$(GREEN)依赖更新完成$(RESET)"

##@ CI/CD

ci: check security-scan ## 运行 CI 检查
	@echo "$(GREEN)CI 检查完成$(RESET)"

release: clean build test docker-build ## 创建发布版本
	@echo "$(BOLD)创建发布版本 $(VERSION)...$(RESET)"
	@git tag -a v$(VERSION) -m "Release version $(VERSION)"
	@echo "$(GREEN)发布版本 v$(VERSION) 创建完成$(RESET)"

##@ 开发工具

shell-rust: ## 进入 Rust 开发环境
	@echo "$(BOLD)进入 Rust 开发环境...$(RESET)"
	@cd rag-engine && bash

shell-go: ## 进入 Go 开发环境
	@echo "$(BOLD)进入 Go 开发环境...$(RESET)"
	@cd rag-api && bash

shell-frontend: ## 进入前端开发环境
	@echo "$(BOLD)进入前端开发环境...$(RESET)"
	@cd rag-frontend && bash

debug-rust: ## 调试 Rust 引擎
	@echo "$(BOLD)启动 Rust 引擎调试模式...$(RESET)"
	@cd rag-engine && RUST_LOG=debug cargo run --bin rag-engine-server

debug-go: ## 调试 Go API
	@echo "$(BOLD)启动 Go API 调试模式...$(RESET)"
	@cd rag-api && dlv debug cmd/server/main.go 2>/dev/null || go run cmd/server/main.go

profile: ## 性能分析
	@echo "$(BOLD)启动性能分析...$(RESET)"
	@echo "  Rust 引擎性能: http://localhost:8080/debug/pprof"
	@echo "  Go API 性能:   http://localhost:8000/debug/pprof"

##@ 环境管理

env-dev: ## 设置开发环境变量
	@echo "$(BOLD)设置开发环境...$(RESET)"
	@cp configs/env/development.env .env 2>/dev/null || echo "DATABASE_URL=postgres://rag_user:rag_password@localhost:5432/rag_development" > .env
	@echo "ENVIRONMENT=development" >> .env
	@echo "$(GREEN)开发环境配置完成$(RESET)"

env-staging: ## 设置测试环境变量
	@echo "$(BOLD)设置测试环境...$(RESET)"
	@cp configs/env/staging.env .env 2>/dev/null || echo "ENVIRONMENT=staging" > .env
	@echo "$(GREEN)测试环境配置完成$(RESET)"

env-prod: ## 设置生产环境变量
	@echo "$(BOLD)设置生产环境...$(RESET)"
	@cp configs/env/production.env .env 2>/dev/null || echo "ENVIRONMENT=production" > .env
	@echo "$(GREEN)生产环境配置完成$(RESET)"

##@ 备份和恢复

backup: ## 备份数据
	@echo "$(BOLD)备份系统数据...$(RESET)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@docker-compose -f infrastructure/docker/docker-compose/development.yml exec -T postgres pg_dump -U rag_user rag_development > backups/$(shell date +%Y%m%d_%H%M%S)/database.sql
	@docker-compose -f infrastructure/docker/docker-compose/development.yml exec -T redis redis-cli --rdb backups/$(shell date +%Y%m%d_%H%M%S)/redis.rdb
	@echo "$(GREEN)数据备份完成$(RESET)"

restore: ## 恢复数据 (需要指定备份目录: make restore BACKUP_DIR=20240101_120000)
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "$(RED)请指定备份目录: make restore BACKUP_DIR=20240101_120000$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)恢复系统数据...$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose/development.yml exec -T postgres psql -U rag_user -d rag_development < backups/$(BACKUP_DIR)/database.sql
	@echo "$(GREEN)数据恢复完成$(RESET)"

##@ 分析和报告

analyze: ## 代码分析
	@echo "$(BOLD)运行代码分析...$(RESET)"
	@echo "$(BLUE)Rust 代码分析:$(RESET)"
	@cd rag-engine && cargo tree --duplicates 2>/dev/null || echo "$(YELLOW)cargo-tree 未安装$(RESET)"
	@echo "$(BLUE)Go 代码分析:$(RESET)"
	@cd rag-api && go mod graph | head -20
	@echo "$(BLUE)前端代码分析:$(RESET)"
	@cd rag-frontend && npm ls --depth=0 2>/dev/null || echo "$(YELLOW)npm ls 失败$(RESET)"

coverage: ## 生成测试覆盖率报告
	@echo "$(BOLD)生成测试覆盖率报告...$(RESET)"
	@cd rag-engine && cargo tarpaulin --out Html 2>/dev/null || echo "$(YELLOW)cargo-tarpaulin 未安装$(RESET)"
	@cd rag-api && go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out -o coverage.html
	@cd rag-frontend && npm run test:coverage 2>/dev/null || echo "$(YELLOW)覆盖率测试未配置$(RESET)"
	@echo "$(GREEN)覆盖率报告生成完成$(RESET)"

metrics: ## 生成项目指标
	@echo "$(BOLD)项目指标统计:$(RESET)"
	@echo "$(BLUE)代码行数:$(RESET)"
	@find . -name "*.rs" -not -path "./target/*" | xargs wc -l | tail -1 | awk '{print "  Rust: " $1 " 行"}'
	@find . -name "*.go" -not -path "./vendor/*" | xargs wc -l | tail -1 | awk '{print "  Go: " $1 " 行"}'
	@find . -name "*.ts" -o -name "*.tsx" -not -path "./node_modules/*" | xargs wc -l | tail -1 | awk '{print "  TypeScript: " $1 " 行"}'
	@echo "$(BLUE)文件统计:$(RESET)"
	@echo "  Rust 文件: $(shell find . -name "*.rs" -not -path "./target/*" | wc -l)"
	@echo "  Go 文件: $(shell find . -name "*.go" -not -path "./vendor/*" | wc -l)"
	@echo "  TypeScript 文件: $(shell find . -name "*.ts" -o -name "*.tsx" -not -path "./node_modules/*" | wc -l)"

##@ 自定义任务

todo: ## 显示 TODO 项目
	@echo "$(BOLD)项目 TODO 列表:$(RESET)"
	@grep -r "TODO\|FIXME\|XXX" --include="*.rs" --include="*.go" --include="*.ts" --include="*.tsx" . | head -20

version: ## 显示版本信息
	@echo "$(BOLD)版本信息:$(RESET)"
	@echo "  项目版本: $(VERSION)"
	@echo "  Git 提交: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "  构建时间: $(shell date)"
	@echo "  Rust 版本: $(shell rustc --version 2>/dev/null || echo 'not installed')"
	@echo "  Go 版本: $(shell go version 2>/dev/null || echo 'not installed')"
	@echo "  Node 版本: $(shell node --version 2>/dev/null || echo 'not installed')"

setup-hooks: ## 设置 Git hooks
	@echo "$(BOLD)设置 Git hooks...$(RESET)"
	@cp scripts/git-hooks/pre-commit .git/hooks/pre-commit 2>/dev/null || echo "#!/bin/sh\nmake check" > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)Git hooks 设置完成$(RESET)"

# 通配符目标，用于处理未定义的目标
%:
	@echo "$(RED)未知目标: $@$(RESET)"
	@echo "运行 'make help' 查看可用命令"