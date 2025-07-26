//! # RAG 引擎服务器
//!
//! RAG 引擎的主要可执行文件，负责启动和管理整个系统。

use clap::{Arg, Command};
use color_eyre::Result;
use rag_engine::{RagEngine, RagConfig};
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use tokio::signal;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

#[tokio::main]
async fn main() -> Result<()> {
    // 设置 color-eyre 用于更好的错误显示
    color_eyre::install()?;

    // 解析命令行参数
    let matches = Command::new("rag-engine-server")
        .version(env!("CARGO_PKG_VERSION"))
        .author("RAG Team <team@rag.com>")
        .about("High-performance RAG engine server")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Sets a custom config file")
                .default_value("config/local.toml"),
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Sets the log level")
                .default_value("info")
                .value_parser(["trace", "debug", "info", "warn", "error"]),
        )
        .arg(
            Arg::new("log-format")
                .long("log-format")
                .value_name("FORMAT")
                .help("Sets the log format")
                .default_value("json")
                .value_parser(["json", "pretty"]),
        )
        .arg(
            Arg::new("bind-http")
                .long("bind-http")
                .value_name("ADDRESS")
                .help("HTTP server bind address")
                .default_value("0.0.0.0:8080"),
        )
        .arg(
            Arg::new("bind-grpc")
                .long("bind-grpc")
                .value_name("ADDRESS")
                .help("gRPC server bind address")
                .default_value("0.0.0.0:9090"),
        )
        .arg(
            Arg::new("workers")
                .short('w')
                .long("workers")
                .value_name("NUMBER")
                .help("Number of worker threads")
                .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            Arg::new("validate-config")
                .long("validate-config")
                .help("Validate configuration and exit")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("check-health")
                .long("check-health")
                .help("Perform health check and exit")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // 获取命令行参数
    let config_path = matches.get_one::<String>("config").unwrap();
    let log_level = matches.get_one::<String>("log-level").unwrap();
    let log_format = matches.get_one::<String>("log-format").unwrap();
    let bind_http = matches.get_one::<String>("bind-http");
    let bind_grpc = matches.get_one::<String>("bind-grpc");
    let workers = matches.get_one::<u32>("workers");
    let validate_config = matches.get_flag("validate-config");
    let check_health = matches.get_flag("check-health");

    // 初始化日志系统
    init_logging(log_level, log_format)?;

    info!("🚀 启动 RAG 引擎服务器 v{}", env!("CARGO_PKG_VERSION"));
    info!("📁 配置文件: {}", config_path);

    // 加载配置
    let mut config = load_config(config_path).await?;

    // 应用命令行参数覆盖
    apply_cli_overrides(&mut config, bind_http, bind_grpc, workers);

    // 如果只是验证配置，则验证后退出
    if validate_config {
        return validate_configuration(&config).await;
    }

    // 如果是健康检查，则检查后退出
    if check_health {
        return perform_health_check(&config).await;
    }

    // 创建并启动 RAG 引擎
    let engine = RagEngine::with_config(config).await?;

    info!("✅ RAG 引擎初始化完成");

    // 启动引擎
    engine.start().await?;

    info!("🎯 RAG 引擎服务已启动，等待请求...");

    // 等待关闭信号
    wait_for_shutdown_signal().await;

    info!("🛑 收到关闭信号，正在优雅关闭...");

    // 优雅关闭引擎
    if let Err(e) = engine.stop().await {
        error!("关闭引擎时发生错误: {}", e);
    }

    info!("✅ RAG 引擎已成功关闭");

    Ok(())
}

/// 初始化日志系统
fn init_logging(level: &str, format: &str) -> Result<()> {
    let level_filter = match level {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    let registry = tracing_subscriber::registry();

    match format {
        "json" => {
            // JSON 格式日志 (生产环境)
            registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .json()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_current_span(false)
                        .with_filter(
                            tracing_subscriber::filter::LevelFilter::from_level(level_filter)
                        ),
                )
                .init();
        }
        "pretty" => {
            // 美化格式日志 (开发环境)
            registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .pretty()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_filter(
                            tracing_subscriber::filter::LevelFilter::from_level(level_filter)
                        ),
                )
                .init();
        }
        _ => {
            return Err(color_eyre::eyre::eyre!("不支持的日志格式: {}", format));
        }
    }

    Ok(())
}

/// 加载配置文件
async fn load_config(config_path: &str) -> Result<RagConfig> {
    let path = PathBuf::from(config_path);

    if !path.exists() {
        warn!("配置文件不存在: {}, 使用默认配置", config_path);
        return Ok(RagConfig::default());
    }

    info!("从配置文件加载配置: {}", config_path);

    match RagConfig::from_file(&path).await {
        Ok(config) => {
            info!("✅ 配置加载成功");
            Ok(config)
        }
        Err(e) => {
            error!("❌ 配置加载失败: {}", e);
            warn!("📋 使用默认配置");
            Ok(RagConfig::default())
        }
    }
}

/// 应用命令行参数覆盖配置
fn apply_cli_overrides(
    config: &mut RagConfig,
    bind_http: Option<&String>,
    bind_grpc: Option<&String>,
    workers: Option<&u32>,
) {
    if let Some(addr) = bind_http {
        info!("🔧 覆盖 HTTP 绑定地址: {}", addr);
        config.network.http.bind_address = addr.clone();
    }

    if let Some(addr) = bind_grpc {
        info!("🔧 覆盖 gRPC 绑定地址: {}", addr);
        config.network.grpc.bind_address = addr.clone();
    }

    if let Some(w) = workers {
        info!("🔧 覆盖工作线程数: {}", w);
        config.concurrency.worker_threads = *w;
    }

    // 从环境变量覆盖敏感配置
    if let Ok(database_url) = env::var("DATABASE_URL") {
        info!("🔧 从环境变量覆盖数据库 URL");
        config.database.postgres.url = database_url;
    }

    if let Ok(redis_url) = env::var("REDIS_URL") {
        info!("🔧 从环境变量覆盖 Redis URL");
        config.cache.redis.url = redis_url;
    }

    if let Ok(qdrant_url) = env::var("QDRANT_URL") {
        info!("🔧 从环境变量覆盖 Qdrant URL");
        config.database.vector.qdrant.url = qdrant_url;
    }

    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        info!("🔧 从环境变量设置 OpenAI API 密钥");
        config.embedding.providers.openai.api_key = openai_key.clone();
        config.llm.providers.openai.api_key = openai_key;
    }

    if let Ok(jaeger_endpoint) = env::var("JAEGER_ENDPOINT") {
        info!("🔧 从环境变量覆盖 Jaeger 端点");
        config.observability.tracing.jaeger_endpoint = jaeger_endpoint;
    }
}

/// 验证配置
async fn validate_configuration(config: &RagConfig) -> Result<()> {
    info!("🔍 验证配置...");

    match config.validate() {
        Ok(_) => {
            info!("✅ 配置验证通过");
            println!("Configuration is valid ✓");
        }
        Err(e) => {
            error!("❌ 配置验证失败: {}", e);
            eprintln!("Configuration validation failed: {}", e);
            std::process::exit(1);
        }
    }

    // 显示配置摘要
    print_config_summary(config);

    Ok(())
}

/// 执行健康检查
async fn perform_health_check(config: &RagConfig) -> Result<()> {
    info!("🏥 执行健康检查...");

    let mut healthy = true;

    // 检查数据库连接
    if let Err(e) = check_database_connection(&config.database.postgres.url).await {
        error!("❌ 数据库连接失败: {}", e);
        eprintln!("Database connection failed: {}", e);
        healthy = false;
    } else {
        info!("✅ 数据库连接正常");
        println!("Database connection ✓");
    }

    // 检查 Redis 连接
    if let Err(e) = check_redis_connection(&config.cache.redis.url).await {
        error!("❌ Redis 连接失败: {}", e);
        eprintln!("Redis connection failed: {}", e);
        healthy = false;
    } else {
        info!("✅ Redis 连接正常");
        println!("Redis connection ✓");
    }

    // 检查 Qdrant 连接
    if let Err(e) = check_qdrant_connection(&config.database.vector.qdrant.url).await {
        error!("❌ Qdrant 连接失败: {}", e);
        eprintln!("Qdrant connection failed: {}", e);
        healthy = false;
    } else {
        info!("✅ Qdrant 连接正常");
        println!("Qdrant connection ✓");
    }

    if healthy {
        info!("✅ 所有健康检查通过");
        println!("All health checks passed ✓");
    } else {
        error!("❌ 部分健康检查失败");
        eprintln!("Some health checks failed ✗");
        std::process::exit(1);
    }

    Ok(())
}

/// 检查数据库连接
async fn check_database_connection(database_url: &str) -> Result<()> {
    use sqlx::postgres::PgPoolOptions;

    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect(database_url)
        .await?;

    sqlx::query("SELECT 1")
        .fetch_one(&pool)
        .await?;

    pool.close().await;
    Ok(())
}

/// 检查 Redis 连接
async fn check_redis_connection(redis_url: &str) -> Result<()> {
    use redis::AsyncCommands;

    let client = redis::Client::open(redis_url)?;
    let mut conn = client.get_async_connection().await?;

    let _: String = conn.ping().await?;

    Ok(())
}

/// 检查 Qdrant 连接
async fn check_qdrant_connection(qdrant_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("{}/health", qdrant_url))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(color_eyre::eyre::eyre!("Qdrant 健康检查失败: {}", response.status()))
    }
}

/// 打印配置摘要
fn print_config_summary(config: &RagConfig) {
    println!("\n📋 Configuration Summary:");
    println!("  App:");
    println!("    Name: {}", config.app.name);
    println!("    Version: {}", config.app.version);
    println!("    Environment: {}", config.app.environment);
    println!("    Debug: {}", config.app.debug);

    println!("  Network:");
    println!("    HTTP: {} (enabled: {})", config.network.http.bind_address, config.network.http.enabled);
    println!("    gRPC: {} (enabled: {})", config.network.grpc.bind_address, config.network.grpc.enabled);
    println!("    WebSocket: {} (enabled: {})", config.network.websocket.bind_address, config.network.websocket.enabled);

    println!("  Database:");
    println!("    PostgreSQL: {} (max_conn: {})", mask_url(&config.database.postgres.url), config.database.postgres.max_connections);
    println!("    Qdrant: {}", config.database.vector.qdrant.url);

    println!("  Cache:");
    println!("    Redis: {} (pool_size: {})", mask_url(&config.cache.redis.url), config.cache.redis.pool_size);
    println!("    Memory: {} MB", config.cache.memory.max_size / 1024 / 1024);

    println!("  Embedding:");
    println!("    Default Provider: {}", config.embedding.default_provider);
    println!("    Batch Size: {}", config.embedding.batch.size);

    println!("  Retrieval:");
    println!("    Default Strategy: {}", config.retrieval.default_strategy);
    println!("    Default Top-K: {}", config.retrieval.default_top_k);

    println!("  LLM:");
    println!("    Default Provider: {}", config.llm.default_provider);

    println!("  Concurrency:");
    println!("    Worker Threads: {}", config.concurrency.worker_threads);
    println!("    Embedding Concurrency: {}", config.concurrency.semaphores.embedding_concurrency);
    println!("    Retrieval Concurrency: {}", config.concurrency.semaphores.retrieval_concurrency);
    println!("    LLM Concurrency: {}", config.concurrency.semaphores.llm_concurrency);

    println!("  Observability:");
    println!("    Metrics: {} ({})", config.observability.metrics.enabled, config.observability.metrics.prometheus_address);
    println!("    Tracing: {} ({})", config.observability.tracing.enabled, config.observability.tracing.jaeger_endpoint);
    println!("    Log Level: {}", config.observability.logging.level);

    println!("  Plugins:");
    println!("    Plugin Dir: {}", config.plugins.plugin_dir);
    println!("    WASM Enabled: {}", config.plugins.enable_wasm);
}

/// 掩码敏感的 URL 信息
fn mask_url(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        let mut masked = parsed.clone();
        if parsed.password().is_some() {
            let _ = masked.set_password(Some("***"));
        }
        if parsed.username() != "" {
            let _ = masked.set_username("***");
        }
        masked.to_string()
    } else {
        "***".to_string()
    }
}

/// 等待关闭信号
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate()).expect("创建 SIGTERM 信号处理器失败");
        let mut sigint = signal(SignalKind::interrupt()).expect("创建 SIGINT 信号处理器失败");
        let mut sigquit = signal(SignalKind::quit()).expect("创建 SIGQUIT 信号处理器失败");

        tokio::select! {
            _ = sigterm.recv() => {
                info!("收到 SIGTERM 信号");
            }
            _ = sigint.recv() => {
                info!("收到 SIGINT 信号 (Ctrl+C)");
            }
            _ = sigquit.recv() => {
                info!("收到 SIGQUIT 信号");
            }
        }
    }

    #[cfg(windows)]
    {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("收到 Ctrl+C 信号");
            }
            Err(err) => {
                error!("无法监听 Ctrl+C 信号: {}", err);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_url() {
        let url_with_password = "postgres://user:password@localhost:5432/db";
        let masked = mask_url(url_with_password);
        assert!(!masked.contains("password"));
        assert!(masked.contains("***"));

        let url_without_password = "http://localhost:8080";
        let masked_simple = mask_url(url_without_password);
        assert_eq!(masked_simple, url_without_password);
    }

    #[test]
    fn test_config_loading() {
        let config = RagConfig::default();
        assert_eq!(config.app.name, "rag-engine");
        assert_eq!(config.app.version, "0.1.0");
        assert!(config.network.http.enabled);
        assert!(config.network.grpc.enabled);
    }
}