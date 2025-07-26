package config

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/viper"
)

// Config 应用主配置结构
type Config struct {
	App        AppConfig        `mapstructure:"app" json:"app"`
	Server     ServerConfig     `mapstructure:"server" json:"server"`
	Database   DatabaseConfig   `mapstructure:"database" json:"database"`
	Redis      RedisConfig      `mapstructure:"redis" json:"redis"`
	RagEngine  RagEngineConfig  `mapstructure:"rag_engine" json:"rag_engine"`
	Auth       AuthConfig       `mapstructure:"auth" json:"auth"`
	Storage    StorageConfig    `mapstructure:"storage" json:"storage"`
	Queue      QueueConfig      `mapstructure:"queue" json:"queue"`
	Monitoring MonitoringConfig `mapstructure:"monitoring" json:"monitoring"`
	Security   SecurityConfig   `mapstructure:"security" json:"security"`
}

// AppConfig 应用配置
type AppConfig struct {
	Name        string `mapstructure:"name" json:"name"`
	Version     string `mapstructure:"version" json:"version"`
	Environment string `mapstructure:"environment" json:"environment"`
	Debug       bool   `mapstructure:"debug" json:"debug"`
	TimeZone    string `mapstructure:"timezone" json:"timezone"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
	Port            string          `mapstructure:"port" json:"port"`
	Host            string          `mapstructure:"host" json:"host"`
	ReadTimeout     time.Duration   `mapstructure:"read_timeout" json:"read_timeout"`
	WriteTimeout    time.Duration   `mapstructure:"write_timeout" json:"write_timeout"`
	IdleTimeout     time.Duration   `mapstructure:"idle_timeout" json:"idle_timeout"`
	ShutdownTimeout int             `mapstructure:"shutdown_timeout" json:"shutdown_timeout"`
	MaxRequestSize  int64           `mapstructure:"max_request_size" json:"max_request_size"`
	CORS            CORSConfig      `mapstructure:"cors" json:"cors"`
	RateLimit       RateLimitConfig `mapstructure:"rate_limit" json:"rate_limit"`
}

// CORSConfig CORS 配置
type CORSConfig struct {
	AllowOrigins     []string `mapstructure:"allow_origins" json:"allow_origins"`
	AllowMethods     []string `mapstructure:"allow_methods" json:"allow_methods"`
	AllowHeaders     []string `mapstructure:"allow_headers" json:"allow_headers"`
	ExposeHeaders    []string `mapstructure:"expose_headers" json:"expose_headers"`
	AllowCredentials bool     `mapstructure:"allow_credentials" json:"allow_credentials"`
	MaxAge           int      `mapstructure:"max_age" json:"max_age"`
}

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	Enabled        bool          `mapstructure:"enabled" json:"enabled"`
	RequestsPerMin int           `mapstructure:"requests_per_min" json:"requests_per_min"`
	BurstSize      int           `mapstructure:"burst_size" json:"burst_size"`
	CleanupWindow  time.Duration `mapstructure:"cleanup_window" json:"cleanup_window"`
}

// DatabaseConfig 数据库配置
type DatabaseConfig struct {
	Host            string        `mapstructure:"host" json:"host"`
	Port            int           `mapstructure:"port" json:"port"`
	Username        string        `mapstructure:"username" json:"username"`
	Password        string        `mapstructure:"password" json:"password"`
	Database        string        `mapstructure:"database" json:"database"`
	SSLMode         string        `mapstructure:"ssl_mode" json:"ssl_mode"`
	MaxOpenConns    int           `mapstructure:"max_open_conns" json:"max_open_conns"`
	MaxIdleConns    int           `mapstructure:"max_idle_conns" json:"max_idle_conns"`
	ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime" json:"conn_max_lifetime"`
	ConnMaxIdleTime time.Duration `mapstructure:"conn_max_idle_time" json:"conn_max_idle_time"`
}

// RedisConfig Redis 配置
type RedisConfig struct {
	Host         string        `mapstructure:"host" json:"host"`
	Port         int           `mapstructure:"port" json:"port"`
	Password     string        `mapstructure:"password" json:"password"`
	Database     int           `mapstructure:"database" json:"database"`
	PoolSize     int           `mapstructure:"pool_size" json:"pool_size"`
	MinIdleConns int           `mapstructure:"min_idle_conns" json:"min_idle_conns"`
	MaxRetries   int           `mapstructure:"max_retries" json:"max_retries"`
	DialTimeout  time.Duration `mapstructure:"dial_timeout" json:"dial_timeout"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout" json:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout" json:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout" json:"idle_timeout"`
}

// RagEngineConfig RAG 引擎配置
type RagEngineConfig struct {
	GRPCAddress    string        `mapstructure:"grpc_address" json:"grpc_address"`
	ConnTimeout    time.Duration `mapstructure:"conn_timeout" json:"conn_timeout"`
	RequestTimeout time.Duration `mapstructure:"request_timeout" json:"request_timeout"`
	MaxRetries     int           `mapstructure:"max_retries" json:"max_retries"`
	KeepAlive      time.Duration `mapstructure:"keep_alive" json:"keep_alive"`
	TLS            TLSConfig     `mapstructure:"tls" json:"tls"`
}

// TLSConfig TLS 配置
type TLSConfig struct {
	Enabled    bool   `mapstructure:"enabled" json:"enabled"`
	CertFile   string `mapstructure:"cert_file" json:"cert_file"`
	KeyFile    string `mapstructure:"key_file" json:"key_file"`
	CAFile     string `mapstructure:"ca_file" json:"ca_file"`
	ServerName string `mapstructure:"server_name" json:"server_name"`
}

// AuthConfig 认证配置
type AuthConfig struct {
	JWT     JWTConfig     `mapstructure:"jwt" json:"jwt"`
	OAuth   OAuthConfig   `mapstructure:"oauth" json:"oauth"`
	RBAC    RBACConfig    `mapstructure:"rbac" json:"rbac"`
	Session SessionConfig `mapstructure:"session" json:"session"`
}

// JWTConfig JWT 配置
type JWTConfig struct {
	SecretKey       string        `mapstructure:"secret_key" json:"secret_key"`
	Issuer          string        `mapstructure:"issuer" json:"issuer"`
	Audience        string        `mapstructure:"audience" json:"audience"`
	AccessTokenTTL  time.Duration `mapstructure:"access_token_ttl" json:"access_token_ttl"`
	RefreshTokenTTL time.Duration `mapstructure:"refresh_token_ttl" json:"refresh_token_ttl"`
	Algorithm       string        `mapstructure:"algorithm" json:"algorithm"`
}

// OAuthConfig OAuth 配置
type OAuthConfig struct {
	Google    OAuthProvider `mapstructure:"google" json:"google"`
	GitHub    OAuthProvider `mapstructure:"github" json:"github"`
	Microsoft OAuthProvider `mapstructure:"microsoft" json:"microsoft"`
}

// OAuthProvider OAuth 提供商配置
type OAuthProvider struct {
	Enabled      bool     `mapstructure:"enabled" json:"enabled"`
	ClientID     string   `mapstructure:"client_id" json:"client_id"`
	ClientSecret string   `mapstructure:"client_secret" json:"client_secret"`
	RedirectURL  string   `mapstructure:"redirect_url" json:"redirect_url"`
	Scopes       []string `mapstructure:"scopes" json:"scopes"`
}

// RBACConfig 基于角色的访问控制配置
type RBACConfig struct {
	Enabled      bool          `mapstructure:"enabled" json:"enabled"`
	CacheEnabled bool          `mapstructure:"cache_enabled" json:"cache_enabled"`
	CacheTTL     time.Duration `mapstructure:"cache_ttl" json:"cache_ttl"`
	DefaultRole  string        `mapstructure:"default_role" json:"default_role"`
}

// SessionConfig 会话配置
type SessionConfig struct {
	CookieName   string        `mapstructure:"cookie_name" json:"cookie_name"`
	CookieDomain string        `mapstructure:"cookie_domain" json:"cookie_domain"`
	CookiePath   string        `mapstructure:"cookie_path" json:"cookie_path"`
	MaxAge       time.Duration `mapstructure:"max_age" json:"max_age"`
	Secure       bool          `mapstructure:"secure" json:"secure"`
	HttpOnly     bool          `mapstructure:"http_only" json:"http_only"`
	SameSite     string        `mapstructure:"same_site" json:"same_site"`
}

// StorageConfig 存储配置
type StorageConfig struct {
	Type  string       `mapstructure:"type" json:"type"`
	Local LocalStorage `mapstructure:"local" json:"local"`
	MinIO MinIOConfig  `mapstructure:"minio" json:"minio"`
	AWS   AWSConfig    `mapstructure:"aws" json:"aws"`
}

// LocalStorage 本地存储配置
type LocalStorage struct {
	BasePath  string `mapstructure:"base_path" json:"base_path"`
	MaxSize   int64  `mapstructure:"max_size" json:"max_size"`
	URLPrefix string `mapstructure:"url_prefix" json:"url_prefix"`
}

// MinIOConfig MinIO 配置
type MinIOConfig struct {
	Endpoint  string `mapstructure:"endpoint" json:"endpoint"`
	AccessKey string `mapstructure:"access_key" json:"access_key"`
	SecretKey string `mapstructure:"secret_key" json:"secret_key"`
	Bucket    string `mapstructure:"bucket" json:"bucket"`
	Region    string `mapstructure:"region" json:"region"`
	UseSSL    bool   `mapstructure:"use_ssl" json:"use_ssl"`
	URLPrefix string `mapstructure:"url_prefix" json:"url_prefix"`
}

// AWSConfig AWS S3 配置
type AWSConfig struct {
	Region    string `mapstructure:"region" json:"region"`
	AccessKey string `mapstructure:"access_key" json:"access_key"`
	SecretKey string `mapstructure:"secret_key" json:"secret_key"`
	Bucket    string `mapstructure:"bucket" json:"bucket"`
	URLPrefix string `mapstructure:"url_prefix" json:"url_prefix"`
	Endpoint  string `mapstructure:"endpoint" json:"endpoint"`
}

// QueueConfig 队列配置
type QueueConfig struct {
	Type     string         `mapstructure:"type" json:"type"`
	Redis    RedisQueue     `mapstructure:"redis" json:"redis"`
	RabbitMQ RabbitMQConfig `mapstructure:"rabbitmq" json:"rabbitmq"`
}

// RedisQueue Redis 队列配置
type RedisQueue struct {
	KeyPrefix         string        `mapstructure:"key_prefix" json:"key_prefix"`
	MaxRetries        int           `mapstructure:"max_retries" json:"max_retries"`
	RetryDelay        time.Duration `mapstructure:"retry_delay" json:"retry_delay"`
	VisibilityTimeout time.Duration `mapstructure:"visibility_timeout" json:"visibility_timeout"`
}

// RabbitMQConfig RabbitMQ 配置
type RabbitMQConfig struct {
	URL            string        `mapstructure:"url" json:"url"`
	Exchange       string        `mapstructure:"exchange" json:"exchange"`
	ExchangeType   string        `mapstructure:"exchange_type" json:"exchange_type"`
	RoutingKey     string        `mapstructure:"routing_key" json:"routing_key"`
	QueueName      string        `mapstructure:"queue_name" json:"queue_name"`
	Durable        bool          `mapstructure:"durable" json:"durable"`
	AutoDelete     bool          `mapstructure:"auto_delete" json:"auto_delete"`
	Exclusive      bool          `mapstructure:"exclusive" json:"exclusive"`
	NoWait         bool          `mapstructure:"no_wait" json:"no_wait"`
	PrefetchCount  int           `mapstructure:"prefetch_count" json:"prefetch_count"`
	ReconnectDelay time.Duration `mapstructure:"reconnect_delay" json:"reconnect_delay"`
}

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	Metrics MetricsConfig `mapstructure:"metrics" json:"metrics"`
	Tracing TracingConfig `mapstructure:"tracing" json:"tracing"`
	Health  HealthConfig  `mapstructure:"health" json:"health"`
	Logging LoggingConfig `mapstructure:"logging" json:"logging"`
}

// MetricsConfig 指标配置
type MetricsConfig struct {
	Enabled   bool   `mapstructure:"enabled" json:"enabled"`
	Path      string `mapstructure:"path" json:"path"`
	Port      string `mapstructure:"port" json:"port"`
	Namespace string `mapstructure:"namespace" json:"namespace"`
	Subsystem string `mapstructure:"subsystem" json:"subsystem"`
}

// TracingConfig 追踪配置
type TracingConfig struct {
	Enabled        bool    `mapstructure:"enabled" json:"enabled"`
	ServiceName    string  `mapstructure:"service_name" json:"service_name"`
	JaegerEndpoint string  `mapstructure:"jaeger_endpoint" json:"jaeger_endpoint"`
	SampleRate     float64 `mapstructure:"sample_rate" json:"sample_rate"`
	Environment    string  `mapstructure:"environment" json:"environment"`
}

// HealthConfig 健康检查配置
type HealthConfig struct {
	Enabled  bool          `mapstructure:"enabled" json:"enabled"`
	Path     string        `mapstructure:"path" json:"path"`
	Interval time.Duration `mapstructure:"interval" json:"interval"`
	Timeout  time.Duration `mapstructure:"timeout" json:"timeout"`
}

// LoggingConfig 日志配置
type LoggingConfig struct {
	Level      string `mapstructure:"level" json:"level"`
	Format     string `mapstructure:"format" json:"format"`
	Output     string `mapstructure:"output" json:"output"`
	FilePath   string `mapstructure:"file_path" json:"file_path"`
	MaxSize    int    `mapstructure:"max_size" json:"max_size"`
	MaxAge     int    `mapstructure:"max_age" json:"max_age"`
	MaxBackups int    `mapstructure:"max_backups" json:"max_backups"`
	Compress   bool   `mapstructure:"compress" json:"compress"`
}

// SecurityConfig 安全配置
type SecurityConfig struct {
	Encryption          EncryptionConfig `mapstructure:"encryption" json:"encryption"`
	CSP                 CSPConfig        `mapstructure:"csp" json:"csp"`
	HSTS                HSTSConfig       `mapstructure:"hsts" json:"hsts"`
	XFrameOptions       string           `mapstructure:"x_frame_options" json:"x_frame_options"`
	XContentTypeOptions bool             `mapstructure:"x_content_type_options" json:"x_content_type_options"`
	XSSProtection       XSSConfig        `mapstructure:"xss_protection" json:"xss_protection"`
}

// EncryptionConfig 加密配置
type EncryptionConfig struct {
	Key       string `mapstructure:"key" json:"key"`
	Algorithm string `mapstructure:"algorithm" json:"algorithm"`
}

// CSPConfig Content Security Policy 配置
type CSPConfig struct {
	Enabled   bool   `mapstructure:"enabled" json:"enabled"`
	Policy    string `mapstructure:"policy" json:"policy"`
	ReportURI string `mapstructure:"report_uri" json:"report_uri"`
}

// HSTSConfig HTTP Strict Transport Security 配置
type HSTSConfig struct {
	Enabled           bool `mapstructure:"enabled" json:"enabled"`
	MaxAge            int  `mapstructure:"max_age" json:"max_age"`
	IncludeSubDomains bool `mapstructure:"include_sub_domains" json:"include_sub_domains"`
	Preload           bool `mapstructure:"preload" json:"preload"`
}

// XSSConfig XSS Protection 配置
type XSSConfig struct {
	Enabled   bool   `mapstructure:"enabled" json:"enabled"`
	Mode      string `mapstructure:"mode" json:"mode"`
	ReportURI string `mapstructure:"report_uri" json:"report_uri"`
}

// Load 加载配置文件
func Load(configPath string) (*Config, error) {
	viper.SetConfigFile(configPath)

	// 设置环境变量前缀
	viper.SetEnvPrefix("RAG_API")
	viper.AutomaticEnv()

	// 读取配置文件
	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("读取配置文件失败: %w", err)
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("解析配置失败: %w", err)
	}

	// 从环境变量覆盖敏感配置
	overrideFromEnv(&config)

	return &config, nil
}

// overrideFromEnv 从环境变量覆盖配置
func overrideFromEnv(config *Config) {
	if dbURL := os.Getenv("DATABASE_URL"); dbURL != "" {
		// 解析数据库 URL
		// 这里可以添加更复杂的 URL 解析逻辑
		config.Database.Host = "localhost" // 简化示例
	}

	if redisURL := os.Getenv("REDIS_URL"); redisURL != "" {
		config.Redis.Host = "localhost" // 简化示例
	}

	if ragEngineAddr := os.Getenv("RAG_ENGINE_GRPC_URL"); ragEngineAddr != "" {
		config.RagEngine.GRPCAddress = ragEngineAddr
	}

	if jwtSecret := os.Getenv("JWT_SECRET_KEY"); jwtSecret != "" {
		config.Auth.JWT.SecretKey = jwtSecret
	}

	if minioAccessKey := os.Getenv("MINIO_ACCESS_KEY"); minioAccessKey != "" {
		config.Storage.MinIO.AccessKey = minioAccessKey
	}

	if minioSecretKey := os.Getenv("MINIO_SECRET_KEY"); minioSecretKey != "" {
		config.Storage.MinIO.SecretKey = minioSecretKey
	}
}

// Validate 验证配置
func (c *Config) Validate() error {
	if c.App.Name == "" {
		return fmt.Errorf("app.name 不能为空")
	}

	if c.Server.Port == "" {
		return fmt.Errorf("server.port 不能为空")
	}

	if c.Database.Host == "" {
		return fmt.Errorf("database.host 不能为空")
	}

	if c.Database.Username == "" {
		return fmt.Errorf("database.username 不能为空")
	}

	if c.Database.Database == "" {
		return fmt.Errorf("database.database 不能为空")
	}

	if c.RagEngine.GRPCAddress == "" {
		return fmt.Errorf("rag_engine.grpc_address 不能为空")
	}

	if c.Auth.JWT.SecretKey == "" {
		return fmt.Errorf("auth.jwt.secret_key 不能为空")
	}

	// 验证环境
	validEnvs := []string{"development", "staging", "production"}
	found := false
	for _, env := range validEnvs {
		if c.App.Environment == env {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("无效的环境: %s, 支持的环境: %v", c.App.Environment, validEnvs)
	}

	return nil
}

// GetDSN 获取数据库连接字符串
func (c *Config) GetDSN() string {
	return fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Database.Host,
		c.Database.Port,
		c.Database.Username,
		c.Database.Password,
		c.Database.Database,
		c.Database.SSLMode,
	)
}

// GetRedisAddr 获取 Redis 地址
func (c *Config) GetRedisAddr() string {
	return fmt.Sprintf("%s:%d", c.Redis.Host, c.Redis.Port)
}

// IsProduction 判断是否为生产环境
func (c *Config) IsProduction() bool {
	return c.App.Environment == "production"
}

// IsDevelopment 判断是否为开发环境
func (c *Config) IsDevelopment() bool {
	return c.App.Environment == "development"
}
