package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"rag-api/internal/config"
	"rag-api/internal/gateway"
	"rag-api/pkg/logger"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
)

var (
	configPath = flag.String("config", "configs/local.yaml", "配置文件路径")
	logLevel   = flag.String("log-level", "info", "日志级别")
	port       = flag.String("port", "8000", "服务端口")
	version    = flag.Bool("version", false, "显示版本信息")
)

const (
	// AppName 应用名称
	AppName = "rag-api"
	// AppVersion 应用版本
	AppVersion = "0.1.0"
	// BuildTime 构建时间 (在构建时通过 ldflags 注入)
	BuildTime = "unknown"
	// GitCommit Git提交哈希 (在构建时通过 ldflags 注入)
	GitCommit = "unknown"
)

// @title RAG System API
// @version 0.1.0
// @description 高性能 RAG 系统 API 服务
// @termsOfService https://example.com/terms
// @contact.name RAG Team
// @contact.email team@rag.com
// @contact.url https://example.com/support
// @license.name MIT
// @license.url https://opensource.org/licenses/MIT
// @host localhost:8000
// @BasePath /api/v1
// @schemes http https
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description JWT Bearer Token
func main() {
	flag.Parse()

	// 显示版本信息
	if *version {
		printVersion()
		return
	}

	// 初始化日志
	if err := initLogger(*logLevel); err != nil {
		log.Fatalf("初始化日志失败: %v", err)
	}

	logger.Info("🚀 启动 RAG API 服务器",
		"app", AppName,
		"version", AppVersion,
		"build_time", BuildTime,
		"git_commit", GitCommit,
	)

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		logger.Fatal("加载配置失败", "error", err, "path", *configPath)
	}

	// 应用命令行参数覆盖
	if *port != "8000" {
		cfg.Server.Port = *port
		logger.Info("覆盖服务端口", "port", *port)
	}

	// 验证配置
	if err := cfg.Validate(); err != nil {
		logger.Fatal("配置验证失败", "error", err)
	}

	logger.Info("✅ 配置加载完成", "environment", cfg.App.Environment)

	// 设置 Gin 模式
	switch cfg.App.Environment {
	case "production":
		gin.SetMode(gin.ReleaseMode)
	case "staging":
		gin.SetMode(gin.TestMode)
	default:
		gin.SetMode(gin.DebugMode)
	}

	// 创建服务器实例
	server, err := gateway.NewServer(cfg)
	if err != nil {
		logger.Fatal("创建服务器失败", "error", err)
	}

	// 启动服务器
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 在 goroutine 中启动服务器
	go func() {
		if err := server.Start(ctx); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatal("服务器启动失败", "error", err)
		}
	}()

	logger.Info("🎯 RAG API 服务器已启动",
		"port", cfg.Server.Port,
		"environment", cfg.App.Environment,
		"debug", cfg.App.Debug,
	)

	// 等待中断信号
	waitForShutdown(ctx, server, cfg)
}

// initLogger 初始化日志系统
func initLogger(level string) error {
	logConfig := logger.Config{
		Level:      level,
		Format:     "json",
		Output:     "stdout",
		TimeFormat: time.RFC3339,
		Caller:     true,
	}

	if gin.Mode() == gin.DebugMode {
		logConfig.Format = "console"
		logConfig.Color = true
	}

	return logger.Init(logConfig)
}

// printVersion 打印版本信息
func printVersion() {
	fmt.Printf("%s version %s\n", AppName, AppVersion)
	fmt.Printf("Build time: %s\n", BuildTime)
	fmt.Printf("Git commit: %s\n", GitCommit)
}

// waitForShutdown 等待关闭信号并优雅关闭
func waitForShutdown(_ context.Context, server *gateway.Server, cfg *config.Config) {
	// 创建信号通道
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	// 等待信号
	sig := <-quit
	logger.Info("🛑 收到关闭信号", "signal", sig.String())

	// 创建关闭上下文
	shutdownCtx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(cfg.Server.ShutdownTimeout)*time.Second,
	)
	defer cancel()

	// 优雅关闭服务器
	logger.Info("🔄 开始优雅关闭...")

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("服务器关闭失败", "error", err)
		return
	}

	logger.Info("✅ RAG API 服务器已优雅关闭")
}
