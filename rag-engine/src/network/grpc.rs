//! # gRPC 服务器模块
//!
//! 提供高性能的 gRPC API 接口，主要用于服务间通信

use std::{net::SocketAddr, sync::Arc, time::Duration};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, error, debug};

use crate::{
    config::RagConfig,
    error::{RagError, RagResult},
    services::rag_service::RagService,
    types::*,
};

// 这里需要生成的 protobuf 代码
// 为了演示，我们使用一个简化的实现

/// gRPC 服务定义 (简化版本)
#[derive(Debug)]
pub struct RagGrpcService {
    rag_service: Arc<RagService>,
}

impl RagGrpcService {
    pub fn new(rag_service: Arc<RagService>) -> Self {
        Self { rag_service }
    }
}

// 这里应该是从 .proto 文件生成的代码
// 为了演示，我们使用简化的实现

/// 搜索请求 (protobuf 消息)
#[derive(Debug, Clone)]
pub struct SearchGrpcRequest {
    pub query: String,
    pub top_k: u32,
    pub workspace_id: Option<String>,
}

/// 搜索响应 (protobuf 消息)
#[derive(Debug, Clone)]
pub struct SearchGrpcResponse {
    pub results: Vec<SearchResultGrpc>,
    pub total_time_ms: u64,
}

/// 搜索结果项 (protobuf 消息)
#[derive(Debug, Clone)]
pub struct SearchResultGrpc {
    pub chunk_id: String,
    pub content: String,
    pub score: f32,
    pub document_id: String,
}

/// 健康检查请求
#[derive(Debug, Clone)]
pub struct HealthCheckGrpcRequest {}

/// 健康检查响应
#[derive(Debug, Clone)]
pub struct HealthCheckGrpcResponse {
    pub status: String,
    pub timestamp: String,
}

// 实际的 trait 实现应该由 tonic 生成
// 这里是简化的手动实现
impl RagGrpcService {
    /// 执行搜索
    pub async fn search(
        &self,
        request: Request<SearchGrpcRequest>,
    ) -> Result<Response<SearchGrpcResponse>, Status> {
        let req = request.into_inner();
        debug!("gRPC 搜索请求: {}", req.query);

        // 转换为内部查询格式
        let query = Query {
            id: uuid::Uuid::new_v4(),
            text: req.query,
            options: QueryOptions {
                strategy: "hybrid".to_string(),
                top_k: req.top_k,
                similarity_threshold: None,
                filters: Vec::new(),
                enable_reranking: false,
                rerank_top_k: None,
                workspace_id: req.workspace_id.and_then(|id| id.parse().ok()),
            },
            timestamp: chrono::Utc::now(),
        };

        // 执行搜索
        match self.rag_service.search(query).await {
            Ok(search_result) => {
                let grpc_results: Vec<SearchResultGrpc> = search_result
                    .results
                    .into_iter()
                    .map(|item| SearchResultGrpc {
                        chunk_id: item.chunk.id.to_string(),
                        content: item.chunk.content,
                        score: item.score,
                        document_id: item.chunk.document_id.to_string(),
                    })
                    .collect();

                let response = SearchGrpcResponse {
                    results: grpc_results,
                    total_time_ms: search_result.metadata.total_time_ms,
                };

                Ok(Response::new(response))
            }
            Err(e) => {
                error!("gRPC 搜索失败: {}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }

    /// 健康检查
    pub async fn health_check(
        &self,
        _request: Request<HealthCheckGrpcRequest>,
    ) -> Result<Response<HealthCheckGrpcResponse>, Status> {
        match self.rag_service.health_check().await {
            Ok(_) => {
                let response = HealthCheckGrpcResponse {
                    status: "healthy".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("健康检查失败: {}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }
}

/// gRPC 服务器
pub struct GrpcServer {
    config: Arc<RagConfig>,
    rag_service: Arc<RagService>,
}

impl GrpcServer {
    /// 创建新的 gRPC 服务器
    pub async fn new(
        config: Arc<RagConfig>,
        rag_service: Arc<RagService>,
    ) -> RagResult<Self> {
        Ok(Self {
            config,
            rag_service,
        })
    }

    /// 启动 gRPC 服务器
    pub async fn serve(self) -> RagResult<()> {
        let addr: SocketAddr = self.config.network.grpc.bind_address
            .parse()
            .map_err(|e| RagError::NetworkError(
                crate::error::NetworkError::ServerBindingFailed {
                    address: self.config.network.grpc.bind_address.clone(),
                }
            ))?;

        info!("🌐 启动 gRPC 服务器: {}", addr);

        // 创建服务实例
        let rag_service = RagGrpcService::new(self.rag_service.clone());

        // 构建服务器
        let server = Server::builder()
            .timeout(Duration::from_secs(self.config.network.grpc.connect_timeout))
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .tcp_nodelay(true)
            .max_concurrent_streams(Some(1000))
            .initial_stream_window_size(Some(1024 * 1024)) // 1MB
            .initial_connection_window_size(Some(1024 * 1024 * 10)) // 10MB
            .max_frame_size(Some(16 * 1024 * 1024)); // 16MB

        // 在实际实现中，这里应该添加由 tonic 生成的服务
        // server.add_service(RagServiceServer::new(rag_service))

        // 简化的服务器启动
        info!("✅ gRPC 服务器配置完成");

        // 实际应该是:
        // server.serve(addr).await.map_err(|e| {
        //     RagError::NetworkError(crate::error::NetworkError::RequestHandlingFailed(
        //         e.to_string(),
        //     ))
        // })?;

        // 简化实现：只是等待
        tokio::time::sleep(Duration::from_secs(1)).await;

        info!("gRPC 服务器已启动 (简化实现)");
        Ok(())
    }
}

/// gRPC 拦截器 - 用于认证、日志、指标等
pub struct GrpcInterceptor {
    config: Arc<RagConfig>,
}

impl GrpcInterceptor {
    pub fn new(config: Arc<RagConfig>) -> Self {
        Self { config }
    }

    /// 认证拦截器
    pub fn auth_interceptor(
        &self,
        req: Request<()>,
    