//! # 配置模块
//!
//! 负责管理 RAG 引擎的所有配置，支持多种配置源：
//! - 配置文件 (TOML/YAML)
//! - 环境变量
//! - 命令行参数

use serde::{Deserialize, Serialize};
use std::path::Path;
use figment::{Figment, providers::{Format, Toml, Env}};
use crate::error::{RagError, RagResult};

/// RAG 引擎主配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// 应用配置
    pub app: AppConfig,
    /// 数据库配置
    pub database: DatabaseConfig,
    /// 缓存配置
    pub cache: CacheConfig,
    /// 网络配置
    pub network: NetworkConfig,
    /// 嵌入配置
    pub embedding: EmbeddingConfig,
    /// 检索配置
    pub retrieval: RetrievalConfig,
    /// LLM 配置
    pub llm: LlmConfig,
    /// 并发配置
    pub concurrency: ConcurrencyConfig,
    /// 可观测性配置
    pub observability: ObservabilityConfig,
    /// 插件配置
    pub plugins: PluginConfig,
}

/// 应用配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// 应用名称
    pub name: String,
    /// 应用版本
    pub version: String,
    /// 运行环境 (development, staging, production)
    pub environment: String,
    /// 调试模式
    pub debug: bool,
    /// 工作目录
    pub work_dir: String,
    /// 临时目录
    pub temp_dir: String,
}

/// 数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// PostgresSQL 配置
    pub postgres: PostgresConfig,
    /// 向量数据库配置
    pub vector: VectorDbConfig,
}

/// PostgresSQL 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    /// 数据库 URL
    pub url: String,
    /// 最大连接数
    pub max_connections: u32,
    /// 最小连接数
    pub min_connections: u32,
    /// 连接超时时间 (秒)
    pub connect_timeout: u64,
    /// 空闲连接超时时间 (秒)
    pub idle_timeout: u64,
    /// 连接最大生命周期 (秒)
    pub max_lifetime: u64,
}

/// 向量数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    /// 提供商 (qdrant, pinecone, weaviate)
    pub provider: String,
    /// Qdrant 配置
    pub qdrant: QdrantConfig,
}

/// Qdrant 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// 服务器 URL
    pub url: String,
    /// API 密钥
    pub api_key: Option<String>,
    /// 超时时间 (秒)
    pub timeout: u64,
    /// 重试次数
    pub max_retries: u32,
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Redis 配置
    pub redis: RedisConfig,
    /// 内存缓存配置
    pub memory: MemoryCacheConfig,
}

/// Redis 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis URL
    pub url: String,
    /// 连接池大小
    pub pool_size: u32,
    /// 最小空闲连接数
    pub min_idle: u32,
    /// 连接超时时间 (秒)
    pub connect_timeout: u64,
    /// 命令超时时间 (秒)
    pub command_timeout: u64,
    /// 重试次数
    pub max_retries: u32,
}

/// 内存缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCacheConfig {
    /// 最大缓存大小 (字节)
    pub max_size: u64,
    /// 默认 TTL (秒)
    pub default_ttl: u64,
    /// 清理间隔 (秒)
    pub cleanup_interval: u64,
}

/// 网络配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// HTTP 服务配置
    pub http: HttpConfig,
    /// gRPC 服务配置
    pub grpc: GrpcConfig,
    /// WebSocket 配置
    pub websocket: WebSocketConfig,
}

/// HTTP 服务配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// 是否启用
    pub enabled: bool,
    /// 绑定地址
    pub bind_address: String,
    /// 请求超时时间 (秒)
    pub request_timeout: u64,
    /// 最大请求体大小 (字节)
    pub max_body_size: u64,
    /// CORS 配置
    pub cors: CorsConfig,
    /// 限流配置
    pub rate_limit: RateLimitConfig,
}

/// CORS 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// 允许的源
    pub allowed_origins: Vec<String>,
    /// 允许的方法
    pub allowed_methods: Vec<String>,
    /// 允许的头部
    pub allowed_headers: Vec<String>,
    /// 是否允许凭证
    pub allow_credentials: bool,
}

/// 限流配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// 是否启用
    pub enabled: bool,
    /// 每秒请求数
    pub requests_per_second: u32,
    /// 突发请求数
    pub burst_size: u32,
}

/// gRPC 服务配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// 是否启用
    pub enabled: bool,
    /// 绑定地址
    pub bind_address: String,
    /// 最大接收消息大小 (字节)
    pub max_receive_message_size: u32,
    /// 最大发送消息大小 (字节)
    pub max_send_message_size: u32,
    /// 连接超时时间 (秒)
    pub connect_timeout: u64,
}

/// WebSocket 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// 是否启用
    pub enabled: bool,
    /// 绑定地址
    pub bind_address: String,
    /// 最大连接数
    pub max_connections: u32,
    /// 心跳间隔 (秒)
    pub heartbeat_interval: u64,
    /// 消息缓冲区大小
    pub message_buffer_size: u32,
}

/// 嵌入配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// 默认提供商
    pub default_provider: String,
    /// 提供商配置
    pub providers: EmbeddingProvidersConfig,
    /// 批处理配置
    pub batch: BatchConfig,
    /// 缓存配置
    pub cache: EmbeddingCacheConfig,
}

/// 嵌入提供商配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProvidersConfig {
    /// 本地模型配置
    pub local: LocalEmbeddingConfig,
    /// OpenAI 配置
    pub openai: OpenAIEmbeddingConfig,
    /// Hugging Face 配置
    pub huggingface: HuggingFaceEmbeddingConfig,
}

/// 本地嵌入配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalEmbeddingConfig {
    /// 模型名称
    pub model_name: String,
    /// 模型路径
    pub model_path: String,
    /// 设备 (cpu, cuda, mps)
    pub device: String,
    /// 批处理大小
    pub batch_size: u32,
    /// 最大序列长度
    pub max_length: u32,
}

/// OpenAI 嵌入配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingConfig {
    /// API 密钥
    pub api_key: String,
    /// 模型名称
    pub model: String,
    /// API 基础 URL
    pub base_url: Option<String>,
    /// 请求超时时间 (秒)
    pub timeout: u64,
    /// 最大重试次数
    pub max_retries: u32,
}

/// Hugging Face 嵌入配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceEmbeddingConfig {
    /// API 密钥
    pub api_key: String,
    /// 模型名称
    pub model: String,
    /// API 基础 URL
    pub base_url: Option<String>,
    /// 请求超时时间 (秒)
    pub timeout: u64,
}

/// 批处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// 批处理大小
    pub size: u32,
    /// 批处理超时时间 (毫秒)
    pub timeout_ms: u64,
    /// 最大队列长度
    pub max_queue_size: u32,
}

/// 嵌入缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheConfig {
    /// 是否启用
    pub enabled: bool,
    /// TTL (秒)
    pub ttl: u64,
    /// 最大缓存条目数
    pub max_entries: u32,
}

/// 检索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// 默认策略
    pub default_strategy: String,
    /// 默认 top-k
    pub default_top_k: u32,
    /// 策略配置
    pub strategies: RetrievalStrategiesConfig,
    /// 重排序配置
    pub reranking: RerankingConfig,
    /// 融合配置
    pub fusion: FusionConfig,
}

/// 检索策略配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalStrategiesConfig {
    /// 密集检索配置
    pub dense: DenseRetrievalConfig,
    /// 稀疏检索配置
    pub sparse: SparseRetrievalConfig,
    /// 混合检索配置
    pub hybrid: HybridRetrievalConfig,
}

/// 密集检索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseRetrievalConfig {
    /// 相似度阈值
    pub similarity_threshold: f32,
    /// 搜索参数
    pub search_params: DenseSearchParams,
}

/// 密集搜索参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseSearchParams {
    /// 候选集大小
    pub ef: u32,
    /// 精确搜索
    pub exact: bool,
}

/// 稀疏检索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseRetrievalConfig {
    /// BM25 参数
    pub bm25: BM25Config,
}

/// BM25 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// k1 参数
    pub k1: f32,
    /// b 参数
    pub b: f32,
}

/// 混合检索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRetrievalConfig {
    /// 密集检索权重
    pub dense_weight: f32,
    /// 稀疏检索权重
    pub sparse_weight: f32,
}

/// 重排序配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingConfig {
    /// 是否启用
    pub enabled: bool,
    /// 重排序模型
    pub model: String,
    /// 重排序 top-k
    pub top_k: u32,
}

/// 融合配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// 融合方法 (rrf, weighted, learned)
    pub method: String,
    /// RRF 参数
    pub rrf_k: f32,
}

/// LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// 默认提供商
    pub default_provider: String,
    /// 提供商配置
    pub providers: LlmProvidersConfig,
    /// 对话配置
    pub conversation: ConversationConfig,
    /// 生成配置
    pub generation: GenerationConfig,
}

/// LLM 提供商配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProvidersConfig {
    /// OpenAI 配置
    pub openai: OpenAILlmConfig,
    /// Anthropic 配置
    pub anthropic: AnthropicLlmConfig,
    /// 本地模型配置
    pub local: LocalLlmConfig,
}

/// OpenAI LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAILlmConfig {
    /// API 密钥
    pub api_key: String,
    /// 模型名称
    pub model: String,
    /// API 基础 URL
    pub base_url: Option<String>,
    /// 请求超时时间 (秒)
    pub timeout: u64,
}

/// Anthropic LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicLlmConfig {
    /// API 密钥
    pub api_key: String,
    /// 模型名称
    pub model: String,
    /// 请求超时时间 (秒)
    pub timeout: u64,
}

/// 本地 LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLlmConfig {
    /// 模型路径
    pub model_path: String,
    /// 设备
    pub device: String,
    /// 最大上下文长度
    pub max_context_length: u32,
}

/// 对话配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// 最大历史长度
    pub max_history_length: u32,
    /// 上下文窗口大小
    pub context_window_size: u32,
    /// 记忆策略
    pub memory_strategy: String,
}

/// 生成配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// 默认温度
    pub default_temperature: f32,
    /// 最大 token 数
    pub max_tokens: u32,
    /// 流式输出
    pub streaming: bool,
    /// 超时时间 (秒)
    pub timeout: u64,
}

/// 并发配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// 工作线程数
    pub worker_threads: u32,
    /// 任务队列配置
    pub task_queue: TaskQueueConfig,
    /// 信号量配置
    pub semaphores: SemaphoreConfig,
}

/// 任务队列配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQueueConfig {
    /// 高优先级队列容量
    pub high_priority_capacity: u32,
    /// 中优先级队列容量
    pub medium_priority_capacity: u32,
    /// 低优先级队列容量
    pub low_priority_capacity: u32,
}

/// 信号量配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemaphoreConfig {
    /// 嵌入并发数
    pub embedding_concurrency: u32,
    /// 检索并发数
    pub retrieval_concurrency: u32,
    /// LLM 并发数
    pub llm_concurrency: u32,
}

/// 可观测性配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// 日志配置
    pub logging: LoggingConfig,
    /// 指标配置
    pub metrics: MetricsConfig,
    /// 追踪配置
    pub tracing: TracingConfig,
    /// 健康检查配置
    pub health: HealthConfig,
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// 日志级别
    pub level: String,
    /// 输出格式 (json, pretty)
    pub format: String,
    /// 输出目标 (stdout, file)
    pub output: String,
    /// 日志文件路径
    pub file_path: Option<String>,
    /// 文件轮转大小 (字节)
    pub rotation_size: Option<u64>,
}

/// 指标配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// 是否启用
    pub enabled: bool,
    /// Prometheus 导出地址
    pub prometheus_address: String,
    /// 指标收集间隔 (秒)
    pub collection_interval: u64,
}

/// 追踪配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// 是否启用
    pub enabled: bool,
    /// Jaeger 端点
    pub jaeger_endpoint: String,
    /// 采样率
    pub sample_rate: f64,
}

/// 健康检查配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// 检查间隔 (秒)
    pub check_interval: u64,
    /// 超时时间 (秒)
    pub timeout: u64,
}

/// 插件配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// 插件目录
    pub plugin_dir: String,
    /// 是否启用 WASM 插件
    pub enable_wasm: bool,
    /// WASM 运行时配置
    pub wasm_runtime: WasmRuntimeConfig,
}

/// WASM 运行时配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRuntimeConfig {
    /// 内存限制 (字节)
    pub memory_limit: u64,
    /// 执行超时时间 (秒)
    pub execution_timeout: u64,
    /// 最大实例数
    pub max_instances: u32,
}

impl RagConfig {
    /// 从文件加载配置
    pub async fn from_file<P: AsRef<Path>>(path: P) -> RagResult<Self> {
        let config = Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("RAG_"))
            .extract()
            .map_err(|e| RagError::ConfigError(e.to_string()))?;

        Ok(config)
    }

    /// 从环境变量加载配置
    pub fn from_env() -> RagResult<Self> {
        let config = Figment::new()
            .merge(Env::prefixed("RAG_"))
            .extract()
            .map_err(|e| RagError::ConfigError(e.to_string()))?;

        Ok(config)
    }

    /// 使用默认配置
    pub fn default() -> Self {
        Self {
            app: AppConfig {
                name: "rag-engine".to_string(),
                version: "0.1.0".to_string(),
                environment: "development".to_string(),
                debug: true,
                work_dir: "./data".to_string(),
                temp_dir: "/tmp".to_string(),
            },
            database: DatabaseConfig {
                postgres: PostgresConfig {
                    url: "postgres://rag_user:rag_password@localhost:5432/rag_development".to_string(),
                    max_connections: 20,
                    min_connections: 5,
                    connect_timeout: 30,
                    idle_timeout: 600,
                    max_lifetime: 3600,
                },
                vector: VectorDbConfig {
                    provider: "qdrant".to_string(),
                    qdrant: QdrantConfig {
                        url: "http://localhost:6333".to_string(),
                        api_key: None,
                        timeout: 30,
                        max_retries: 3,
                    },
                },
            },
            cache: CacheConfig {
                redis: RedisConfig {
                    url: "redis://localhost:6379".to_string(),
                    pool_size: 20,
                    min_idle: 5,
                    connect_timeout: 10,
                    command_timeout: 5,
                    max_retries: 3,
                },
                memory: MemoryCacheConfig {
                    max_size: 1024 * 1024 * 1024, // 1GB
                    default_ttl: 3600,
                    cleanup_interval: 300,
                },
            },
            network: NetworkConfig {
                http: HttpConfig {
                    enabled: true,
                    bind_address: "0.0.0.0:8080".to_string(),
                    request_timeout: 30,
                    max_body_size: 16 * 1024 * 1024, // 16MB
                    cors: CorsConfig {
                        allowed_origins: vec!["*".to_string()],
                        allowed_methods: vec!["GET".to_string(), "POST".to_string(), "PUT".to_string(), "DELETE".to_string()],
                        allowed_headers: vec!["*".to_string()],
                        allow_credentials: true,
                    },
                    rate_limit: RateLimitConfig {
                        enabled: true,
                        requests_per_second: 100,
                        burst_size: 10,
                    },
                },
                grpc: GrpcConfig {
                    enabled: true,
                    bind_address: "0.0.0.0:9090".to_string(),
                    max_receive_message_size: 16 * 1024 * 1024,
                    max_send_message_size: 16 * 1024 * 1024,
                    connect_timeout: 30,
                },
                websocket: WebSocketConfig {
                    enabled: true,
                    bind_address: "0.0.0.0:8081".to_string(),
                    max_connections: 1000,
                    heartbeat_interval: 30,
                    message_buffer_size: 1024,
                },
            },
            embedding: EmbeddingConfig {
                default_provider: "local".to_string(),
                providers: EmbeddingProvidersConfig {
                    local: LocalEmbeddingConfig {
                        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                        model_path: "./models/embedding".to_string(),
                        device: "cpu".to_string(),
                        batch_size: 32,
                        max_length: 512,
                    },
                    openai: OpenAIEmbeddingConfig {
                        api_key: "".to_string(),
                        model: "text-embedding-ada-002".to_string(),
                        base_url: None,
                        timeout: 30,
                        max_retries: 3,
                    },
                    huggingface: HuggingFaceEmbeddingConfig {
                        api_key: "".to_string(),
                        model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                        base_url: None,
                        timeout: 30,
                    },
                },
                batch: BatchConfig {
                    size: 32,
                    timeout_ms: 100,
                    max_queue_size: 1000,
                },
                cache: EmbeddingCacheConfig {
                    enabled: true,
                    ttl: 3600,
                    max_entries: 10000,
                },
            },
            retrieval: RetrievalConfig {
                default_strategy: "hybrid".to_string(),
                default_top_k: 10,
                strategies: RetrievalStrategiesConfig {
                    dense: DenseRetrievalConfig {
                        similarity_threshold: 0.7,
                        search_params: DenseSearchParams {
                            ef: 128,
                            exact: false,
                        },
                    },
                    sparse: SparseRetrievalConfig {
                        bm25: BM25Config {
                            k1: 1.2,
                            b: 0.75,
                        },
                    },
                    hybrid: HybridRetrievalConfig {
                        dense_weight: 0.7,
                        sparse_weight: 0.3,
                    },
                },
                reranking: RerankingConfig {
                    enabled: false,
                    model: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
                    top_k: 100,
                },
                fusion: FusionConfig {
                    method: "rrf".to_string(),
                    rrf_k: 60.0,
                },
            },
            llm: LlmConfig {
                default_provider: "openai".to_string(),
                providers: LlmProvidersConfig {
                    openai: OpenAILlmConfig {
                        api_key: "".to_string(),
                        model: "gpt-3.5-turbo".to_string(),
                        base_url: None,
                        timeout: 60,
                    },
                    anthropic: AnthropicLlmConfig {
                        api_key: "".to_string(),
                        model: "claude-3-sonnet-20240229".to_string(),
                        timeout: 60,
                    },
                    local: LocalLlmConfig {
                        model_path: "./models/llm".to_string(),
                        device: "cpu".to_string(),
                        max_context_length: 4096,
                    },
                },
                conversation: ConversationConfig {
                    max_history_length: 10,
                    context_window_size: 4096,
                    memory_strategy: "sliding_window".to_string(),
                },
                generation: GenerationConfig {
                    default_temperature: 0.7,
                    max_tokens: 1000,
                    streaming: true,
                    timeout: 60,
                },
            },
            concurrency: ConcurrencyConfig {
                worker_threads: num_cpus::get() as u32,
                task_queue: TaskQueueConfig {
                    high_priority_capacity: 1000,
                    medium_priority_capacity: 5000,
                    low_priority_capacity: 10000,
                },
                semaphores: SemaphoreConfig {
                    embedding_concurrency: 10,
                    retrieval_concurrency: 20,
                    llm_concurrency: 5,
                },
            },
            observability: ObservabilityConfig {
                logging: LoggingConfig {
                    level: "info".to_string(),
                    format: "json".to_string(),
                    output: "stdout".to_string(),
                    file_path: None,
                    rotation_size: None,
                },
                metrics: MetricsConfig {
                    enabled: true,
                    prometheus_address: "0.0.0.0:9091".to_string(),
                    collection_interval: 60,
                },
                tracing: TracingConfig {
                    enabled: true,
                    jaeger_endpoint: "http://localhost:14268/api/traces".to_string(),
                    sample_rate: 0.1,
                },
                health: HealthConfig {
                    check_interval: 30,
                    timeout: 10,
                },
            },
            plugins: PluginConfig {
                plugin_dir: "./plugins".to_string(),
                enable_wasm: true,
                wasm_runtime: WasmRuntimeConfig {
                    memory_limit: 64 * 1024 * 1024, // 64MB
                    execution_timeout: 30,
                    max_instances: 100,
                },
            },
        }
    }

    /// 验证配置
    pub fn validate(&self) -> RagResult<()> {
        // 验证数据库连接配置
        if self.database.postgres.url.is_empty() {
            return Err(RagError::ConfigError("Database URL cannot be empty".to_string()));
        }

        // 验证网络绑定地址
        if self.network.http.enabled && self.network.http.bind_address.is_empty() {
            return Err(RagError::ConfigError("HTTP bind address cannot be empty".to_string()));
        }

        // 验证并发配置
        if self.concurrency.worker_threads == 0 {
            return Err(RagError::ConfigError("Worker threads must be greater than 0".to_string()));
        }

        Ok(())
    }
}