package logger

import (
	"fmt"
	_ "io"
	"os"
	"time"

	_ "github.com/sirupsen/logrus"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Config 日志配置
type Config struct {
	Level      string `json:"level"`
	Format     string `json:"format"`      // json, console
	Output     string `json:"output"`      // stdout, stderr, file
	FilePath   string `json:"file_path"`   // 文件路径
	TimeFormat string `json:"time_format"` // 时间格式
	Caller     bool   `json:"caller"`      // 是否显示调用者信息
	Color      bool   `json:"color"`       // 是否使用颜色
}

var (
	// 全局 zap logger
	zapLogger *zap.Logger
	// 全局 sugar logger (更方便的接口)
	sugarLogger *zap.SugaredLogger
)

// Init 初始化日志系统
func Init(config Config) error {
	// 解析日志级别
	level, err := parseLogLevel(config.Level)
	if err != nil {
		return fmt.Errorf("invalid log level: %s", config.Level)
	}

	// 配置编码器
	var encoderConfig zapcore.EncoderConfig
	if config.Format == "json" {
		encoderConfig = zap.NewProductionEncoderConfig()
	} else {
		encoderConfig = zap.NewDevelopmentEncoderConfig()
		if config.Color {
			encoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
		}
	}

	// 设置时间格式
	if config.TimeFormat != "" {
		encoderConfig.TimeKey = "timestamp"
		encoderConfig.EncodeTime = zapcore.TimeEncoderOfLayout(config.TimeFormat)
	} else {
		encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	}

	// 设置调用者信息
	if config.Caller {
		encoderConfig.CallerKey = "caller"
		encoderConfig.EncodeCaller = zapcore.ShortCallerEncoder
	}

	// 创建编码器
	var encoder zapcore.Encoder
	if config.Format == "json" {
		encoder = zapcore.NewJSONEncoder(encoderConfig)
	} else {
		encoder = zapcore.NewConsoleEncoder(encoderConfig)
	}

	// 配置输出
	var writer zapcore.WriteSyncer
	switch config.Output {
	case "stdout":
		writer = zapcore.AddSync(os.Stdout)
	case "stderr":
		writer = zapcore.AddSync(os.Stderr)
	case "file":
		if config.FilePath == "" {
			return fmt.Errorf("file path is required when output is file")
		}
		file, err := os.OpenFile(config.FilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return fmt.Errorf("failed to open log file: %w", err)
		}
		writer = zapcore.AddSync(file)
	default:
		writer = zapcore.AddSync(os.Stdout)
	}

	// 创建 core
	core := zapcore.NewCore(encoder, writer, level)

	// 创建 logger
	var options []zap.Option
	if config.Caller {
		options = append(options, zap.AddCaller(), zap.AddCallerSkip(1))
	}
	options = append(options, zap.AddStacktrace(zap.ErrorLevel))

	zapLogger = zap.New(core, options...)
	sugarLogger = zapLogger.Sugar()

	return nil
}

// parseLogLevel 解析日志级别
func parseLogLevel(level string) (zapcore.Level, error) {
	switch level {
	case "debug":
		return zap.DebugLevel, nil
	case "info":
		return zap.InfoLevel, nil
	case "warn", "warning":
		return zap.WarnLevel, nil
	case "error":
		return zap.ErrorLevel, nil
	case "fatal":
		return zap.FatalLevel, nil
	case "panic":
		return zap.PanicLevel, nil
	default:
		return zap.InfoLevel, fmt.Errorf("unknown log level: %s", level)
	}
}

// Sync 同步日志输出
func Sync() {
	if zapLogger != nil {
		_ = zapLogger.Sync()
	}
}

// GetLogger 获取原始 zap logger
func GetLogger() *zap.Logger {
	return zapLogger
}

// GetSugarLogger 获取 sugar logger
func GetSugarLogger() *zap.SugaredLogger {
	return sugarLogger
}

// 便捷的日志方法

// Debug 调试日志
func Debug(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Debugw(msg, fields...)
	}
}

// Info 信息日志
func Info(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Infow(msg, fields...)
	}
}

// Warn 警告日志
func Warn(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Warnw(msg, fields...)
	}
}

// Error 错误日志
func Error(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Errorw(msg, fields...)
	}
}

// Fatal 致命错误日志
func Fatal(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Fatalw(msg, fields...)
	}
}

// Panic panic 日志
func Panic(msg string, fields ...interface{}) {
	if sugarLogger != nil {
		sugarLogger.Panicw(msg, fields...)
	}
}

// 结构化日志方法

// DebugFields 结构化调试日志
func DebugFields(msg string, fields ...zap.Field) {
	if zapLogger != nil {
		zapLogger.Debug(msg, fields...)
	}
}

// InfoFields 结构化信息日志
func InfoFields(msg string, fields ...zap.Field) {
	if zapLogger != nil {
		zapLogger.Info(msg, fields...)
	}
}

// WarnFields 结构化警告日志
func WarnFields(msg string, fields ...zap.Field) {
	if zapLogger != nil {
		zapLogger.Warn(msg, fields...)
	}
}

// ErrorFields 结构化错误日志
func ErrorFields(msg string, fields ...zap.Field) {
	if zapLogger != nil {
		zapLogger.Error(msg, fields...)
	}
}

// FatalFields 结构化致命错误日志
func FatalFields(msg string, fields ...zap.Field) {
	if zapLogger != nil {
		zapLogger.Fatal(msg, fields...)
	}
}

// WithFields 创建带字段的 logger
func WithFields(fields ...zap.Field) *zap.Logger {
	if zapLogger != nil {
		return zapLogger.With(fields...)
	}
	return nil
}

// WithKeyValues 创建带键值对的 sugar logger
func WithKeyValues(keysAndValues ...interface{}) *zap.SugaredLogger {
	if sugarLogger != nil {
		return sugarLogger.With(keysAndValues...)
	}
	return nil
}

// LoggerMiddleware 创建 Gin 中间件
func LoggerMiddleware() func(c interface{}) {
	return func(c interface{}) {
		// 这里应该根据实际的 web 框架实现
		// 简化实现，实际应该是 gin.HandlerFunc
	}
}

// RequestLogger 请求日志记录器
type RequestLogger struct {
	logger *zap.Logger
}

// NewRequestLogger 创建请求日志记录器
func NewRequestLogger() *RequestLogger {
	return &RequestLogger{
		logger: zapLogger,
	}
}

// LogRequest 记录请求日志
func (rl *RequestLogger) LogRequest(
	method, path, clientIP, userAgent string,
	statusCode, bodySize int,
	latency time.Duration,
	fields ...zap.Field,
) {
	allFields := []zap.Field{
		zap.String("method", method),
		zap.String("path", path),
		zap.String("client_ip", clientIP),
		zap.String("user_agent", userAgent),
		zap.Int("status_code", statusCode),
		zap.Int("body_size", bodySize),
		zap.Duration("latency", latency),
	}
	allFields = append(allFields, fields...)

	if statusCode >= 500 {
		rl.logger.Error("HTTP request", allFields...)
	} else if statusCode >= 400 {
		rl.logger.Warn("HTTP request", allFields...)
	} else {
		rl.logger.Info("HTTP request", allFields...)
	}
}

// ErrorLogger 错误日志记录器
type ErrorLogger struct {
	logger *zap.Logger
}

// NewErrorLogger 创建错误日志记录器
func NewErrorLogger() *ErrorLogger {
	return &ErrorLogger{
		logger: zapLogger,
	}
}

// LogError 记录错误日志
func (el *ErrorLogger) LogError(err error, msg string, fields ...zap.Field) {
	allFields := []zap.Field{
		zap.Error(err),
	}
	allFields = append(allFields, fields...)
	el.logger.Error(msg, allFields...)
}

// LogPanic 记录 panic 日志
func (el *ErrorLogger) LogPanic(recovered interface{}, stack []byte, fields ...zap.Field) {
	allFields := []zap.Field{
		zap.Any("panic", recovered),
		zap.ByteString("stack", stack),
	}
	allFields = append(allFields, fields...)
	el.logger.Error("Panic recovered", allFields...)
}

// MetricsLogger 指标日志记录器
type MetricsLogger struct {
	logger *zap.Logger
}

// NewMetricsLogger 创建指标日志记录器
func NewMetricsLogger() *MetricsLogger {
	return &MetricsLogger{
		logger: zapLogger,
	}
}

// LogMetric 记录指标日志
func (ml *MetricsLogger) LogMetric(name string, value float64, tags map[string]string) {
	fields := []zap.Field{
		zap.String("metric_name", name),
		zap.Float64("value", value),
	}

	for k, v := range tags {
		fields = append(fields, zap.String(k, v))
	}

	ml.logger.Info("Metric", fields...)
}

// BusinessLogger 业务日志记录器
type BusinessLogger struct {
	logger *zap.Logger
}

// NewBusinessLogger 创建业务日志记录器
func NewBusinessLogger() *BusinessLogger {
	return &BusinessLogger{
		logger: zapLogger,
	}
}

// LogUserAction 记录用户操作
func (bl *BusinessLogger) LogUserAction(
	userID, action, resource string,
	success bool,
	details map[string]interface{},
) {
	fields := []zap.Field{
		zap.String("user_id", userID),
		zap.String("action", action),
		zap.String("resource", resource),
		zap.Bool("success", success),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range details {
		fields = append(fields, zap.Any(k, v))
	}

	if success {
		bl.logger.Info("User action", fields...)
	} else {
		bl.logger.Warn("User action failed", fields...)
	}
}

// LogDocumentOperation 记录文档操作
func (bl *BusinessLogger) LogDocumentOperation(
	operation, documentID, userID string,
	metadata map[string]interface{},
) {
	fields := []zap.Field{
		zap.String("operation", operation),
		zap.String("document_id", documentID),
		zap.String("user_id", userID),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range metadata {
		fields = append(fields, zap.Any(k, v))
	}

	bl.logger.Info("Document operation", fields...)
}

// LogSearchQuery 记录搜索查询
func (bl *BusinessLogger) LogSearchQuery(
	query, userID string,
	resultCount int,
	latencyMs int64,
	filters map[string]interface{},
) {
	fields := []zap.Field{
		zap.String("query", query),
		zap.String("user_id", userID),
		zap.Int("result_count", resultCount),
		zap.Int64("latency_ms", latencyMs),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range filters {
		fields = append(fields, zap.Any(k, v))
	}

	bl.logger.Info("Search query", fields...)
}

// SecurityLogger 安全日志记录器
type SecurityLogger struct {
	logger *zap.Logger
}

// NewSecurityLogger 创建安全日志记录器
func NewSecurityLogger() *SecurityLogger {
	return &SecurityLogger{
		logger: zapLogger,
	}
}

// LogAuthAttempt 记录认证尝试
func (sl *SecurityLogger) LogAuthAttempt(
	username, clientIP, userAgent string,
	success bool,
	reason string,
) {
	fields := []zap.Field{
		zap.String("username", username),
		zap.String("client_ip", clientIP),
		zap.String("user_agent", userAgent),
		zap.Bool("success", success),
		zap.String("reason", reason),
		zap.Time("timestamp", time.Now()),
	}

	if success {
		sl.logger.Info("Authentication attempt", fields...)
	} else {
		sl.logger.Warn("Authentication failed", fields...)
	}
}

// LogSecurityEvent 记录安全事件
func (sl *SecurityLogger) LogSecurityEvent(
	eventType, userID, clientIP, description string,
	severity string,
	metadata map[string]interface{},
) {
	fields := []zap.Field{
		zap.String("event_type", eventType),
		zap.String("user_id", userID),
		zap.String("client_ip", clientIP),
		zap.String("description", description),
		zap.String("severity", severity),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range metadata {
		fields = append(fields, zap.Any(k, v))
	}

	switch severity {
	case "critical", "high":
		sl.logger.Error("Security event", fields...)
	case "medium":
		sl.logger.Warn("Security event", fields...)
	default:
		sl.logger.Info("Security event", fields...)
	}
}

// PerformanceLogger 性能日志记录器
type PerformanceLogger struct {
	logger *zap.Logger
}

// NewPerformanceLogger 创建性能日志记录器
func NewPerformanceLogger() *PerformanceLogger {
	return &PerformanceLogger{
		logger: zapLogger,
	}
}

// LogSlowQuery 记录慢查询
func (pl *PerformanceLogger) LogSlowQuery(
	query string,
	duration time.Duration,
	threshold time.Duration,
	metadata map[string]interface{},
) {
	if duration < threshold {
		return
	}

	fields := []zap.Field{
		zap.String("query", query),
		zap.Duration("duration", duration),
		zap.Duration("threshold", threshold),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range metadata {
		fields = append(fields, zap.Any(k, v))
	}

	pl.logger.Warn("Slow query detected", fields...)
}

// LogAPIPerformance 记录 API 性能
func (pl *PerformanceLogger) LogAPIPerformance(
	endpoint, method string,
	duration time.Duration,
	statusCode int,
	metadata map[string]interface{},
) {
	fields := []zap.Field{
		zap.String("endpoint", endpoint),
		zap.String("method", method),
		zap.Duration("duration", duration),
		zap.Int("status_code", statusCode),
		zap.Time("timestamp", time.Now()),
	}

	for k, v := range metadata {
		fields = append(fields, zap.Any(k, v))
	}

	// 根据响应时间决定日志级别
	if duration > 5*time.Second {
		pl.logger.Error("Very slow API", fields...)
	} else if duration > 1*time.Second {
		pl.logger.Warn("Slow API", fields...)
	} else {
		pl.logger.Debug("API performance", fields...)
	}
}
