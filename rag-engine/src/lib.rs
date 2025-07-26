//! # RAG Engine
//!
//! 高性能的 RAG (Retrieval-Augmented Generation) 引擎，使用 Rust 构建。
//!
//! ## 功能特性
//!
//! - 🚀 高并发文档处理和检索
//! - 🧠 多种嵌入模型支持 (本地/远程)
//! - 🔍 混合检索策略 (密集+稀疏)
//! - 🤖 LLM 集成和对话管理
//! - 🔌 插件化架构
//! - 📊 完整的可观测性
//!
//! ## 基本使用
//!
//! ```rust,no_run
//! use rag_engine::RagEngine;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let engine = RagEngine::new().await?;
//!     engine.start().await?;
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod error;
pub mod cache;
pub mod types;
// pub mod utils;
//
// // 核心模块
// pub mod concurrency;
// pub mod cache;
// pub mod pipeline;
// // pub mod embedding;
// pub mod retrieval;
// pub mod llm;
// pub mod multimodal;
//
// // 扩展模块
// pub mod plugins;
// pub mod observability;
//
// // 服务模块
// pub mod controllers;
pub mod services;
pub mod network;

// 重新导出核心类型
pub use config::RagConfig;
pub use error::{RagError, RagResult};
pub use services::rag_service::RagService;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error};

/// RAG 引擎的主要结构体
///
/// 这是整个系统的入口点，负责协调各个子系统的运行。
#[derive(Clone)]
pub struct RagEngine {
    /// 配置信息
    config: Arc<RagConfig>,
    /// RAG 服务实例
    service: Arc<RagService>,
    /// 运行状态
    running: Arc<RwLock<bool>>,
}

impl RagEngine {
    /// 创建新的 RAG 引擎实例
    ///
    /// # 错误
    ///
    /// 如果配置加载失败或者服务初始化失败，会返回错误。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use rag_engine::RagEngine;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let engine = RagEngine::new().await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn new() -> RagResult<Self> {
        Self::with_config_path("config/local.toml").await
    }

    /// 使用指定配置文件创建 RAG 引擎实例
    ///
    /// # 参数
    ///
    /// * `config_path` - 配置文件路径
    ///
    /// # 错误
    ///
    /// 如果配置加载失败或者服务初始化失败，会返回错误。
    pub async fn with_config_path(config_path: &str) -> RagResult<Self> {
        info!("🚀 初始化 RAG 引擎...");

        // 加载配置
        let config = Arc::new(RagConfig::from_file(config_path).await?);
        info!("✅ 配置加载完成");

        // 初始化服务
        let service = Arc::new(RagService::new(config.clone()).await?);
        info!("✅ 服务初始化完成");

        Ok(Self {
            config,
            service,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// 使用配置对象创建 RAG 引擎实例
    ///
    /// # 参数
    ///
    /// * `config` - 配置对象
    pub async fn with_config(config: RagConfig) -> RagResult<Self> {
        let config = Arc::new(config);
        let service = Arc::new(RagService::new(config.clone()).await?);

        Ok(Self {
            config,
            service,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// 启动 RAG 引擎
    ///
    /// 这个方法会启动所有的网络服务和后台任务。
    ///
    /// # 错误
    ///
    /// 如果服务启动失败，会返回错误。
    pub async fn start(&self) -> RagResult<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(RagError::AlreadyRunning);
        }

        info!("🚀 启动 RAG 引擎服务...");

        // 启动网络服务
        self.start_network_services().await?;

        // 启动后台任务
        self.start_background_tasks().await?;

        *running = true;
        info!("✅ RAG 引擎启动完成");

        Ok(())
    }

    /// 停止 RAG 引擎
    ///
    /// 优雅地关闭所有服务和后台任务。
    pub async fn stop(&self) -> RagResult<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }

        info!("🛑 停止 RAG 引擎...");

        // 停止后台任务
        self.stop_background_tasks().await?;

        // 停止网络服务
        self.stop_network_services().await?;

        *running = false;
        info!("✅ RAG 引擎已停止");

        Ok(())
    }

    /// 检查引擎是否正在运行
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// 获取配置
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// 获取服务实例
    pub fn service(&self) -> &RagService {
        &self.service
    }

    /// 启动网络服务
    async fn start_network_services(&self) -> RagResult<()> {
        use crate::network::{http, grpc, websocket};

        info!("🌐 启动网络服务...");

        // 启动 HTTP 服务器
        if self.config.network.http.enabled {
            let http_server = http::HttpServer::new(
                self.config.clone(),
                self.service.clone(),
            ).await?;

            tokio::spawn(async move {
                if let Err(e) = http_server.serve().await {
                    error!("HTTP 服务器错误: {}", e);
                }
            });

            info!("✅ HTTP 服务器启动完成: {}", self.config.network.http.bind_address);
        }

        // 启动 gRPC 服务器
        if self.config.network.grpc.enabled {
            let grpc_server = grpc::GrpcServer::new(
                self.config.clone(),
                self.service.clone(),
            ).await?;

            tokio::spawn(async move {
                if let Err(e) = grpc_server.serve().await {
                    error!("gRPC 服务器错误: {}", e);
                }
            });

            info!("✅ gRPC 服务器启动完成: {}", self.config.network.grpc.bind_address);
        }

        // 启动 WebSocket 服务器
        if self.config.network.websocket.enabled {
            let ws_server = websocket::WebSocketServer::new(
                self.config.clone(),
                self.service.clone(),
            ).await?;

            tokio::spawn(async move {
                if let Err(e) = ws_server.serve().await {
                    error!("WebSocket 服务器错误: {}", e);
                }
            });

            info!("✅ WebSocket 服务器启动完成: {}", self.config.network.websocket.bind_address);
        }

        Ok(())
    }

    /// 停止网络服务
    async fn stop_network_services(&self) -> RagResult<()> {
        info!("🌐 停止网络服务...");
        // 实现网络服务的优雅关闭
        Ok(())
    }

    /// 启动后台任务
    async fn start_background_tasks(&self) -> RagResult<()> {
        info!("⚙️ 启动后台任务...");

        // 启动健康检查任务
        self.start_health_check_task().await?;

        // 启动指标收集任务
        self.start_metrics_collection_task().await?;

        // 启动缓存清理任务
        self.start_cache_cleanup_task().await?;

        Ok(())
    }

    /// 停止后台任务
    async fn stop_background_tasks(&self) -> RagResult<()> {
        info!("⚙️ 停止后台任务...");
        // 实现后台任务的优雅关闭
        Ok(())
    }

    /// 启动健康检查任务
    async fn start_health_check_task(&self) -> RagResult<()> {
        let service = self.service.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(30)
            );

            loop {
                interval.tick().await;

                match service.health_check().await {
                    Ok(_) => {
                        tracing::debug!("健康检查通过");
                    }
                    Err(e) => {
                        error!("健康检查失败: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// 启动指标收集任务
    async fn start_metrics_collection_task(&self) -> RagResult<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(60)
            );

            loop {
                interval.tick().await;

                // 收集系统指标
                tracing::debug!("收集系统指标");
            }
        });

        Ok(())
    }

    /// 启动缓存清理任务
    async fn start_cache_cleanup_task(&self) -> RagResult<()> {
        let service = self.service.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(300) // 5 分钟
            );

            loop {
                interval.tick().await;

                if let Err(e) = service.cleanup_cache().await {
                    error!("缓存清理失败: {}", e);
                }
            }
        });

        Ok(())
    }
}

// 确保引擎可以安全地在线程间传递
unsafe impl Send for RagEngine {}
unsafe impl Sync for RagEngine {}