//! # 错误处理模块
//!
//! 定义了 RAG 引擎中所有可能的错误类型，提供统一的错误处理接口。

use thiserror::Error;
use std::fmt;

/// RAG 引擎的结果类型
pub type RagResult<T> = Result<T, RagError>;

/// RAG 引擎的主要错误类型
#[derive(Error, Debug)]
pub enum RagError {
    /// 配置错误
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// 数据库错误
    #[error("Database error: {0}")]
    DatabaseError(#[from] DatabaseError),

    /// 缓存错误
    #[error("Cache error: {0}")]
    CacheError(#[from] CacheError),

    /// 嵌入错误
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    /// 检索错误
    #[error("Retrieval error: {0}")]
    RetrievalError(#[from] RetrievalError),

    /// LLM 错误
    #[error("LLM error: {0}")]
    LlmError(#[from] LlmError),

    /// 管道错误
    #[error("Pipeline error: {0}")]
    PipelineError(#[from] PipelineError),

    /// 网络错误
    #[error("Network error: {0}")]
    NetworkError(#[from] NetworkError),

    /// 插件错误
    #[error("Plugin error: {0}")]
    PluginError(#[from] PluginError),

    /// 并发错误
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// 验证错误
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// 序列化错误
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// I/O 错误
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// 超时错误
    #[error("Timeout error: operation timed out after {timeout}s")]
    TimeoutError { timeout: u64 },

    /// 资源不存在错误
    #[error("Resource not found: {resource_type} with id '{id}'")]
    NotFoundError {
        resource_type: String,
        id: String,
    },

    /// 资源已存在错误
    #[error("Resource already exists: {resource_type} with id '{id}'")]
    AlreadyExistsError {
        resource_type: String,
        id: String,
    },

    /// 权限错误
    #[error("Permission denied: {operation}")]
    PermissionDeniedError { operation: String },

    /// 限流错误
    #[error("Rate limit exceeded: {limit} requests per {window}s")]
    RateLimitError { limit: u32, window: u32 },

    /// 服务不可用错误
    #[error("Service unavailable: {service}")]
    ServiceUnavailableError { service: String },

    /// 引擎已运行错误
    #[error("RAG engine is already running")]
    AlreadyRunning,

    /// 引擎未运行错误
    #[error("RAG engine is not running")]
    NotRunning,

    /// 内部错误
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// 数据库相关错误
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Connection pool exhausted")]
    PoolExhausted,

    #[error("SQLx error: {0}")]
    SqlxError(#[from] sqlx::Error),
}

/// 缓存相关错误
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Redis connection failed: {0}")]
    RedisConnectionFailed(String),

    #[error("Cache operation failed: {0}")]
    OperationFailed(String),

    #[error("Cache miss for key: {key}")]
    CacheMiss { key: String },

    #[error("Cache key expired: {key}")]
    KeyExpired { key: String },

    #[error("Memory cache full")]
    MemoryCacheFull,

    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),
}

/// 嵌入相关错误
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model loading failed: {model}")]
    ModelLoadFailed { model: String },

    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("API request failed: {provider} - {error}")]
    ApiRequestFailed { provider: String, error: String },

    #[error("Invalid model configuration: {0}")]
    InvalidModelConfig(String),

    #[error("Batch processing failed: {0}")]
    BatchProcessingFailed(String),

    #[error("Unsupported provider: {provider}")]
    UnsupportedProvider { provider: String },

    #[error("Model not found: {model}")]
    ModelNotFound { model: String },
}

/// 检索相关错误
#[derive(Error, Debug)]
pub enum RetrievalError {
    #[error("Vector store connection failed: {store}")]
    VectorStoreConnectionFailed { store: String },

    #[error("Search failed: {0}")]
    SearchFailed(String),

    #[error("Index not found: {index}")]
    IndexNotFound { index: String },

    #[error("Index creation failed: {index}")]
    IndexCreationFailed { index: String },

    #[error("Document insertion failed: {0}")]
    DocumentInsertionFailed(String),

    #[error("Query processing failed: {0}")]
    QueryProcessingFailed(String),

    #[error("Reranking failed: {0}")]
    RerankingFailed(String),

    #[error("Fusion failed: {0}")]
    FusionFailed(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Qdrant error: {0}")]
    QdrantError(String),
}

/// LLM 相关错误
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("API request failed: {provider} - {error}")]
    ApiRequestFailed { provider: String, error: String },

    #[error("Model loading failed: {model}")]
    ModelLoadFailed { model: String },

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Prompt processing failed: {0}")]
    PromptProcessingFailed(String),

    #[error("Context limit exceeded: {current} > {limit}")]
    ContextLimitExceeded { current: usize, limit: usize },

    #[error("Invalid response format: {0}")]
    InvalidResponseFormat(String),

    #[error("Streaming failed: {0}")]
    StreamingFailed(String),

    #[error("Conversation management failed: {0}")]
    ConversationFailed(String),

    #[error("Unsupported provider: {provider}")]
    UnsupportedProvider { provider: String },
}

/// 管道相关错误
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Document parsing failed: {file} - {error}")]
    DocumentParsingFailed { file: String, error: String },

    #[error("Text cleaning failed: {0}")]
    TextCleaningFailed(String),

    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),

    #[error("Metadata extraction failed: {0}")]
    MetadataExtractionFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Unsupported file format: {format}")]
    UnsupportedFileFormat { format: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid document: {0}")]
    InvalidDocument(String),
}

/// 网络相关错误
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Server binding failed: {address}")]
    ServerBindingFailed { address: String },

    #[error("Request handling failed: {0}")]
    RequestHandlingFailed(String),

    #[error("Response serialization failed: {0}")]
    ResponseSerializationFailed(String),

    #[error("WebSocket connection failed: {0}")]
    WebSocketConnectionFailed(String),

    #[error("gRPC service failed: {0}")]
    GrpcServiceFailed(String),

    #[error("HTTP client error: {0}")]
    HttpClientError(String),

    #[error("Connection timeout")]
    ConnectionTimeout,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

/// 插件相关错误
#[derive(Error, Debug)]
pub enum PluginError {
    #[error("Plugin loading failed: {plugin}")]
    PluginLoadFailed { plugin: String },

    #[error("Plugin execution failed: {plugin} - {error}")]
    PluginExecutionFailed { plugin: String, error: String },

    #[error("WASM runtime error: {0}")]
    WasmRuntimeError(String),

    #[error("Plugin not found: {plugin}")]
    PluginNotFound { plugin: String },

    #[error("Plugin validation failed: {plugin} - {error}")]
    PluginValidationFailed { plugin: String, error: String },

    #[error("Plugin interface mismatch: {plugin}")]
    PluginInterfaceMismatch { plugin: String },

    #[error("Plugin security violation: {plugin} - {violation}")]
    PluginSecurityViolation { plugin: String, violation: String },

    #[error("Plugin timeout: {plugin}")]
    PluginTimeout { plugin: String },
}

impl RagError {
    /// 检查错误是否为临时性错误（可重试）
    pub fn is_retriable(&self) -> bool {
        match self {
            RagError::TimeoutError { .. } => true,
            RagError::ServiceUnavailableError { .. } => true,
            RagError::NetworkError(NetworkError::ConnectionTimeout) => true,
            RagError::DatabaseError(DatabaseError::PoolExhausted) => true,
            RagError::CacheError(CacheError::RedisConnectionFailed(_)) => true,
            RagError::EmbeddingError(EmbeddingError::ApiRequestFailed { .. }) => true,
            RagError::LlmError(LlmError::ApiRequestFailed { .. }) => true,
            _ => false,
        }
    }

    /// 检查错误是否为客户端错误（4xx 类错误）
    pub fn is_client_error(&self) -> bool {
        match self {
            RagError::ValidationError(_) => true,
            RagError::NotFoundError { .. } => true,
            RagError::PermissionDeniedError { .. } => true,
            RagError::RateLimitError { .. } => true,
            RagError::NetworkError(NetworkError::InvalidRequest(_)) => true,
            RagError::PipelineError(PipelineError::UnsupportedFileFormat { .. }) => true,
            _ => false,
        }
    }

    /// 检查错误是否为服务器错误（5xx 类错误）
    pub fn is_server_error(&self) -> bool {
        match self {
            RagError::InternalError(_) => true,
            RagError::ServiceUnavailableError { .. } => true,
            RagError::DatabaseError(_) => true,
            RagError::CacheError(_) => true,
            _ => false,
        }
    }

    /// 获取错误码
    pub fn error_code(&self) -> &'static str {
        match self {
            RagError::ConfigError(_) => "CONFIG_ERROR",
            RagError::DatabaseError(_) => "DATABASE_ERROR",
            RagError::CacheError(_) => "CACHE_ERROR",
            RagError::EmbeddingError(_) => "EMBEDDING_ERROR",
            RagError::RetrievalError(_) => "RETRIEVAL_ERROR",
            RagError::LlmError(_) => "LLM_ERROR",
            RagError::PipelineError(_) => "PIPELINE_ERROR",
            RagError::NetworkError(_) => "NETWORK_ERROR",
            RagError::PluginError(_) => "PLUGIN_ERROR",
            RagError::ConcurrencyError(_) => "CONCURRENCY_ERROR",
            RagError::ValidationError(_) => "VALIDATION_ERROR",
            RagError::SerializationError(_) => "SERIALIZATION_ERROR",
            RagError::IoError(_) => "IO_ERROR",
            RagError::TimeoutError { .. } => "TIMEOUT_ERROR",
            RagError::NotFoundError { .. } => "NOT_FOUND_ERROR",
            RagError::AlreadyExistsError { .. } => "ALREADY_EXISTS_ERROR",
            RagError::PermissionDeniedError { .. } => "PERMISSION_DENIED_ERROR",
            RagError::RateLimitError { .. } => "RATE_LIMIT_ERROR",
            RagError::ServiceUnavailableError { .. } => "SERVICE_UNAVAILABLE_ERROR",
            RagError::AlreadyRunning => "ALREADY_RUNNING_ERROR",
            RagError::NotRunning => "NOT_RUNNING_ERROR",
            RagError::InternalError(_) => "INTERNAL_ERROR",
        }
    }

    /// 获取 HTTP 状态码
    pub fn http_status_code(&self) -> u16 {
        match self {
            RagError::ValidationError(_) => 400,
            RagError::NotFoundError { .. } => 404,
            RagError::PermissionDeniedError { .. } => 403,
            RagError::RateLimitError { .. } => 429,
            RagError::AlreadyExistsError { .. } => 409,
            RagError::TimeoutError { .. } => 408,
            RagError::ServiceUnavailableError { .. } => 503,
            _ => 500,
        }
    }
}

/// 错误扩展 trait，为错误提供额外的上下文信息
pub trait ErrorExt {
    /// 添加上下文信息
    fn with_context(self, context: &str) -> RagError;

    /// 添加操作信息
    fn with_operation(self, operation: &str) -> RagError;
}

impl<E> ErrorExt for E
where
    E: Into<RagError>,
{
    fn with_context(self, context: &str) -> RagError {
        let error = self.into();
        RagError::InternalError(format!("{}: {}", context, error))
    }

    fn with_operation(self, operation: &str) -> RagError {
        let error = self.into();
        RagError::InternalError(format!("Operation '{}' failed: {}", operation, error))
    }
}

/// 便捷宏，用于创建特定类型的错误
#[macro_export]
macro_rules! rag_error {
    (config, $msg:expr) => {
        RagError::ConfigError($msg.to_string())
    };
    (validation, $msg:expr) => {
        RagError::ValidationError($msg.to_string())
    };
    (not_found, $type:expr, $id:expr) => {
        RagError::NotFoundError {
            resource_type: $type.to_string(),
            id: $id.to_string(),
        }
    };
    (already_exists, $type:expr, $id:expr) => {
        RagError::AlreadyExistsError {
            resource_type: $type.to_string(),
            id: $id.to_string(),
        }
    };
    (timeout, $timeout:expr) => {
        RagError::TimeoutError { timeout: $timeout }
    };
    (internal, $msg:expr) => {
        RagError::InternalError($msg.to_string())
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let config_error = RagError::ConfigError("test".to_string());
        assert_eq!(config_error.error_code(), "CONFIG_ERROR");
        assert_eq!(config_error.http_status_code(), 500);
        assert!(!config_error.is_retriable());
        assert!(config_error.is_server_error());
    }

    #[test]
    fn test_retriable_errors() {
        let timeout_error = RagError::TimeoutError { timeout: 30 };
        assert!(timeout_error.is_retriable());

        let not_found_error = RagError::NotFoundError {
            resource_type: "document".to_string(),
            id: "123".to_string(),
        };
        assert!(!not_found_error.is_retriable());
    }

    #[test]
    fn test_client_errors() {
        let validation_error = RagError::ValidationError("invalid input".to_string());
        assert!(validation_error.is_client_error());
        assert_eq!(validation_error.http_status_code(), 400);
    }

    #[test]
    fn test_error_macro() {
        let error = rag_error!(not_found, "document", "123");
        match error {
            RagError::NotFoundError { resource_type, id } => {
                assert_eq!(resource_type, "document");
                assert_eq!(id, "123");
            }
            _ => panic!("Expected NotFoundError"),
        }
    }
}