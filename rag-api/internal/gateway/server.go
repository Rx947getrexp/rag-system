package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"rag-api/internal/config"
	"rag-api/internal/handlers"
	"rag-api/pkg/logger"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/gzip"
	"github.com/gin-contrib/requestid"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server HTTP 服务器
type Server struct {
	config     *config.Config
	httpServer *http.Server
	engine     *gin.Engine
}

// NewServer 创建新的服务器实例
func NewServer(cfg *config.Config) (*Server, error) {
	// 设置 Gin 模式
	switch cfg.App.Environment {
	case "production":
		gin.SetMode(gin.ReleaseMode)
	case "staging":
		gin.SetMode(gin.TestMode)
	default:
		gin.SetMode(gin.DebugMode)
	}

	// 创建 Gin 引擎
	engine := gin.New()

	// 创建服务器实例
	server := &Server{
		config: cfg,
		engine: engine,
	}

	// 设置中间件
	server.setupMiddlewares()

	// 设置路由
	server.setupRoutes()

	// 创建 HTTP 服务器
	server.httpServer = &http.Server{
		Addr:         fmt.Sprintf("%s:%s", cfg.Server.Host, cfg.Server.Port),
		Handler:      engine,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	return server, nil
}

// setupMiddlewares 设置中间件
func (s *Server) setupMiddlewares() {
	// 恢复中间件
	s.engine.Use(gin.Recovery())

	// 请求 ID 中间件
	s.engine.Use(requestid.New())

	// 日志中间件
	s.engine.Use(s.loggerMiddleware())

	// CORS 中间件
	s.engine.Use(s.corsMiddleware())

	// Gzip 压缩中间件
	s.engine.Use(gzip.Gzip(gzip.DefaultCompression))

	// 限流中间件
	if s.config.Server.RateLimit.Enabled {
		s.engine.Use(s.rateLimitMiddleware())
	}

	// 指标中间件
	s.engine.Use(s.metricsMiddleware())

	// 安全中间件
	s.engine.Use(s.securityMiddleware())
}

// setupRoutes 设置路由
func (s *Server) setupRoutes() {
	// 健康检查路由
	s.engine.GET("/health", handlers.HealthCheck)
	s.engine.GET("/ready", handlers.ReadinessCheck)

	// 指标路由
	s.engine.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// API 路由组
	v1 := s.engine.Group("/api/v1")
	{
		// 文档管理
		docs := v1.Group("/documents")
		{
			docs.GET("", handlers.ListDocuments)
			docs.POST("", handlers.UploadDocument)
			docs.GET("/:id", handlers.GetDocument)
			docs.PUT("/:id", handlers.UpdateDocument)
			docs.DELETE("/:id", handlers.DeleteDocument)
			docs.GET("/:id/chunks", handlers.GetDocumentChunks)
			docs.POST("/:id/reindex", handlers.ReindexDocument)
		}

		// 搜索功能
		search := v1.Group("/search")
		{
			search.POST("", handlers.Search)
			search.POST("/suggest", handlers.SearchSuggestions)
			search.POST("/similar", handlers.FindSimilar)
		}

		// 对话功能
		chat := v1.Group("/chat")
		{
			chat.POST("", handlers.ChatCompletion)
			chat.POST("/stream", handlers.ChatStream)
			chat.GET("/conversations", handlers.ListConversations)
			chat.POST("/conversations", handlers.CreateConversation)
			chat.GET("/conversations/:id", handlers.GetConversation)
			chat.DELETE("/conversations/:id", handlers.DeleteConversation)
		}

		// 嵌入功能
		embeddings := v1.Group("/embeddings")
		{
			embeddings.POST("", handlers.GenerateEmbeddings)
			embeddings.POST("/batch", handlers.BatchGenerateEmbeddings)
			embeddings.GET("/models", handlers.ListEmbeddingModels)
		}

		// 工作空间管理
		workspaces := v1.Group("/workspaces")
		{
			workspaces.GET("", handlers.ListWorkspaces)
			workspaces.POST("", handlers.CreateWorkspace)
			workspaces.GET("/:id", handlers.GetWorkspace)
			workspaces.PUT("/:id", handlers.UpdateWorkspace)
			workspaces.DELETE("/:id", handlers.DeleteWorkspace)
			workspaces.GET("/:id/members", handlers.GetWorkspaceMembers)
			workspaces.POST("/:id/members", handlers.AddWorkspaceMember)
			workspaces.DELETE("/:id/members/:user_id", handlers.RemoveWorkspaceMember)
		}

		// 用户管理
		users := v1.Group("/users")
		{
			users.GET("/me", handlers.GetCurrentUser)
			users.PUT("/me", handlers.UpdateCurrentUser)
			users.GET("/me/preferences", handlers.GetUserPreferences)
			users.PUT("/me/preferences", handlers.UpdateUserPreferences)
		}

		// 管理功能
		admin := v1.Group("/admin")
		{
			admin.GET("/stats", handlers.GetSystemStats)
			admin.GET("/health", handlers.GetSystemHealth)
			admin.POST("/cache/clear", handlers.ClearCache)
			admin.GET("/logs", handlers.GetLogs)
		}
	}

	// WebSocket 路由
	s.engine.GET("/ws", handlers.WebSocketHandler)

	// 文件上传路由
	s.engine.POST("/upload", handlers.UploadFile)

	// 静态文件服务 (如果需要)
	if s.config.App.Environment == "development" {
		s.engine.Static("/static", "./static")
	}
}

// loggerMiddleware 日志中间件
func (s *Server) loggerMiddleware() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		// 使用自定义日志格式
		logger.Info("HTTP request",
			"method", param.Method,
			"path", param.Path,
			"status", param.StatusCode,
			"latency", param.Latency,
			"client_ip", param.ClientIP,
			"user_agent", param.Request.UserAgent(),
			"error", param.ErrorMessage,
		)
		return ""
	})
}

// corsMiddleware CORS 中间件
func (s *Server) corsMiddleware() gin.HandlerFunc {
	cfg := cors.Config{
		AllowOrigins:     s.config.Server.CORS.AllowOrigins,
		AllowMethods:     s.config.Server.CORS.AllowMethods,
		AllowHeaders:     s.config.Server.CORS.AllowHeaders,
		ExposeHeaders:    s.config.Server.CORS.ExposeHeaders,
		AllowCredentials: s.config.Server.CORS.AllowCredentials,
		MaxAge:           time.Duration(s.config.Server.CORS.MaxAge) * time.Second,
	}

	return cors.New(cfg)
}

// rateLimitMiddleware 限流中间件
func (s *Server) rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 简化的限流实现
		// 实际应该使用 redis 或其他存储来实现分布式限流

		// 这里可以集成如 github.com/didip/tollbooth 等限流库

		c.Next()
	}
}

// metricsMiddleware 指标中间件
func (s *Server) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		c.Next()

		duration := time.Since(start)

		// 记录请求指标
		// 这里可以集成 Prometheus 指标
		logger.Debug("Request metrics",
			"method", c.Request.Method,
			"path", c.FullPath(),
			"status", c.Writer.Status(),
			"duration_ms", duration.Milliseconds(),
			"size", c.Writer.Size(),
		)
	}
}

// securityMiddleware 安全中间件
func (s *Server) securityMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 设置安全头
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", s.config.Security.XFrameOptions)
		c.Header("X-XSS-Protection", "1; mode=block")

		// CSP 头
		if s.config.Security.CSP.Enabled {
			c.Header("Content-Security-Policy", s.config.Security.CSP.Policy)
		}

		// HSTS 头 (仅 HTTPS)
		if s.config.Security.HSTS.Enabled && c.Request.TLS != nil {
			hstsHeader := fmt.Sprintf("max-age=%d", s.config.Security.HSTS.MaxAge)
			if s.config.Security.HSTS.IncludeSubDomains {
				hstsHeader += "; includeSubDomains"
			}
			if s.config.Security.HSTS.Preload {
				hstsHeader += "; preload"
			}
			c.Header("Strict-Transport-Security", hstsHeader)
		}

		c.Next()
	}
}

// Start 启动服务器
func (s *Server) Start(ctx context.Context) error {
	logger.Info("启动 HTTP 服务器",
		"address", s.httpServer.Addr,
		"environment", s.config.App.Environment,
	)

	// 在 goroutine 中启动服务器
	errChan := make(chan error, 1)
	go func() {
		errChan <- s.httpServer.ListenAndServe()
	}()

	// 等待上下文取消或服务器错误
	select {
	case <-ctx.Done():
		logger.Info("收到停止信号，正在关闭服务器...")
		return s.Shutdown(context.Background())
	case err := <-errChan:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("服务器启动失败", "error", err)
			return err
		}
		return nil
	}
}

// Shutdown 优雅关闭服务器
func (s *Server) Shutdown(ctx context.Context) error {
	logger.Info("正在关闭 HTTP 服务器...")

	// 设置关闭超时
	shutdownCtx, cancel := context.WithTimeout(ctx,
		time.Duration(s.config.Server.ShutdownTimeout)*time.Second)
	defer cancel()

	// 关闭服务器
	if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error("服务器关闭失败", "error", err)
		return err
	}

	logger.Info("HTTP 服务器已优雅关闭")
	return nil
}

// GetEngine 获取 Gin 引擎 (用于测试)
func (s *Server) GetEngine() *gin.Engine {
	return s.engine
}

// GetHTTPServer 获取 HTTP 服务器 (用于测试)
func (s *Server) GetHTTPServer() *http.Server {
	return s.httpServer
}
