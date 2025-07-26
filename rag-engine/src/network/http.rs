//! # HTTP 服务器模块
//!
//! 提供 RESTful API 接口，处理文档上传、检索请求等 HTTP 操作

use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, error, warn};
use uuid::Uuid;

use crate::{
    config::RagConfig,
    error::{RagError, RagResult},
    services::rag_service::RagService,
    types::*,
};

/// HTTP 服务器状态
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RagConfig>,
    pub rag_service: Arc<RagService>,
}

/// HTTP 服务器
pub struct HttpServer {
    config: Arc<RagConfig>,
    rag_service: Arc<RagService>,
}

impl HttpServer {
    /// 创建新的 HTTP 服务器
    pub async fn new(
        config: Arc<RagConfig>,
        rag_service: Arc<RagService>,
    ) -> RagResult<Self> {
        Ok(Self {
            config,
            rag_service,
        })
    }

    /// 启动 HTTP 服务器
    pub async fn serve(self) -> RagResult<()> {
        let addr: SocketAddr = self.config.network.http.bind_address
            .parse()
            .map_err(|e| RagError::NetworkError(
                crate::error::NetworkError::ServerBindingFailed {
                    address: self.config.network.http.bind_address.clone(),
                }
            ))?;

        info!("🌐 启动 HTTP 服务器: {}", addr);

        // 创建应用状态
        let state = AppState {
            config: self.config.clone(),
            rag_service: self.rag_service.clone(),
        };

        // 构建路由
        let app = self.build_router(state);

        // 创建监听器
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            RagError::NetworkError(crate::error::NetworkError::ServerBindingFailed {
                address: addr.to_string(),
            })
        })?;

        // 启动服务器
        axum::serve(listener, app).await.map_err(|e| {
            RagError::NetworkError(crate::error::NetworkError::RequestHandlingFailed(
                e.to_string(),
            ))
        })?;

        Ok(())
    }

    /// 构建路由
    fn build_router(&self, state: AppState) -> Router {
        Router::new()
            // 健康检查
            .route("/health", get(health_check))
            .route("/ready", get(readiness_check))

            // API v1 路由
            .nest("/api/v1", self.build_api_v1_routes())

            // 指标端点
            .route("/metrics", get(metrics))

            // 调试端点 (仅开发环境)
            .route("/debug/config", get(debug_config))
            .route("/debug/stats", get(debug_stats))

            // 中间件
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CompressionLayer::new())
                    .layer(TimeoutLayer::new(
                        std::time::Duration::from_secs(
                            self.config.network.http.request_timeout
                        )
                    ))
                    .layer(CorsLayer::permissive()) // 开发环境使用宽松的 CORS
            )
            .with_state(state)
    }

    /// 构建 API v1 路由
    fn build_api_v1_routes(&self) -> Router<AppState> {
        Router::new()
            // 文档管理
            .nest("/documents", self.build_document_routes())

            // 检索功能
            .nest("/search", self.build_search_routes())

            // 对话功能
            .nest("/chat", self.build_chat_routes())

            // 嵌入功能
            .nest("/embeddings", self.build_embedding_routes())

            // 工作空间管理
            .nest("/workspaces", self.build_workspace_routes())
    }

    /// 文档管理路由
    fn build_document_routes(&self) -> Router<AppState> {
        Router::new()
            .route("/", get(list_documents).post(upload_document))
            .route("/:id", get(get_document).put(update_document).delete(delete_document))
            .route("/:id/chunks", get(get_document_chunks))
            .route("/:id/reindex", post(reindex_document))
            .route("/batch", post(batch_upload_documents))
    }

    /// 检索功能路由
    fn build_search_routes(&self) -> Router<AppState> {
        Router::new()
            .route("/", post(search_documents))
            .route("/suggest", post(search_suggestions))
            .route("/similar", post(find_similar))
            .route("/hybrid", post(hybrid_search))
    }

    /// 对话功能路由
    fn build_chat_routes(&self) -> Router<AppState> {
        Router::new()
            .route("/", post(chat_completion))
            .route("/stream", post(chat_stream))
            .route("/conversations", get(list_conversations).post(create_conversation))
            .route("/conversations/:id", get(get_conversation).delete(delete_conversation))
            .route("/conversations/:id/messages", get(get_conversation_messages))
    }

    /// 嵌入功能路由
    fn build_embedding_routes(&self) -> Router<AppState> {
        Router::new()
            .route("/", post(generate_embeddings))
            .route("/batch", post(batch_generate_embeddings))
            .route("/models", get(list_embedding_models))
    }

    /// 工作空间管理路由
    fn build_workspace_routes(&self) -> Router<AppState> {
        Router::new()
            .route("/", get(list_workspaces).post(create_workspace))
            .route("/:id", get(get_workspace).put(update_workspace).delete(delete_workspace))
            .route("/:id/members", get(get_workspace_members).post(add_workspace_member))
            .route("/:id/members/:user_id", delete(remove_workspace_member))
    }
}

// ============================================================================
// 处理器函数 (Handlers)
// ============================================================================

/// 健康检查
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    match state.rag_service.health_check().await {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now(),
            "version": env!("CARGO_PKG_VERSION")
        }))),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "unhealthy",
            "error": e.to_string(),
            "timestamp": chrono::Utc::now()
        })))
    }
}

/// 就绪检查
async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    // 检查各个组件是否就绪
    let mut checks = HashMap::new();

    // 检查数据库连接
    checks.insert("database", state.rag_service.check_database().await.is_ok());

    // 检查向量数据库
    checks.insert("vector_db", state.rag_service.check_vector_db().await.is_ok());

    // 检查缓存
    checks.insert("cache", state.rag_service.check_cache().await.is_ok());

    let all_ready = checks.values().all(|&ready| ready);
    let status_code = if all_ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (status_code, Json(serde_json::json!({
        "status": if all_ready { "ready" } else { "not_ready" },
        "checks": checks,
        "timestamp": chrono::Utc::now()
    })))
}

/// 指标端点
async fn metrics(State(_state): State<AppState>) -> impl IntoResponse {
    // 这里应该返回 Prometheus 格式的指标
    // 简化版本，实际应该使用 prometheus crate
    (StatusCode::OK, "# HELP rag_requests_total Total number of requests\n")
}

/// 调试配置
async fn debug_config(State(state): State<AppState>) -> impl IntoResponse {
    if !state.config.app.debug {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Debug endpoints are disabled in production"
        })));
    }

    (StatusCode::OK, Json(serde_json::json!({
        "app": {
            "name": state.config.app.name,
            "version": state.config.app.version,
            "environment": state.config.app.environment
        },
        "network": {
            "http_enabled": state.config.network.http.enabled,
            "grpc_enabled": state.config.network.grpc.enabled
        }
    })))
}

/// 调试统计
async fn debug_stats(State(state): State<AppState>) -> impl IntoResponse {
    if !state.config.app.debug {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Debug endpoints are disabled in production"
        })));
    }

    match state.rag_service.get_system_stats().await {
        Ok(stats) => (StatusCode::OK, Json(stats)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": e.to_string()
        })))
    }
}

// ============================================================================
// 文档管理处理器
// ============================================================================

/// 文档查询参数
#[derive(Debug, Deserialize)]
struct DocumentQuery {
    page: Option<u32>,
    page_size: Option<u32>,
    workspace_id: Option<Uuid>,
    category: Option<String>,
    tags: Option<String>,
}

/// 列出文档
async fn list_documents(
    State(state): State<AppState>,
    Query(params): Query<DocumentQuery>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1);
    let page_size = params.page_size.unwrap_or(20).min(100); // 最大100个

    match state.rag_service.list_documents(page, page_size, params.workspace_id).await {
        Ok(documents) => (StatusCode::OK, Json(ApiResponse::success(documents))),
        Err(e) => {
            error!("列出文档失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<Vec<Document>>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 上传文档请求
#[derive(Debug, Deserialize)]
struct UploadDocumentRequest {
    title: Option<String>,
    workspace_id: Option<Uuid>,
    tags: Option<Vec<String>>,
    category: Option<String>,
}

/// 上传文档
async fn upload_document(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut request: UploadDocumentRequest = UploadDocumentRequest {
        title: None,
        workspace_id: None,
        tags: None,
        category: None,
    };

    // 解析 multipart 数据
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let field_name = field.name().unwrap_or("").to_string();

        match field_name.as_str() {
            "file" => {
                filename = field.file_name().map(|s| s.to_string());
                file_data = Some(field.bytes().await.unwrap_or_default().to_vec());
            }
            "title" => {
                request.title = field.text().await.ok();
            }
            "workspace_id" => {
                if let Ok(text) = field.text().await {
                    request.workspace_id = text.parse().ok();
                }
            }
            "category" => {
                request.category = field.text().await.ok();
            }
            "tags" => {
                if let Ok(text) = field.text().await {
                    request.tags = serde_json::from_str(&text).ok();
                }
            }
            _ => {}
        }
    }

    // 验证必需字段
    let file_data = match file_data {
        Some(data) => data,
        None => {
            return (StatusCode::BAD_REQUEST, Json(ApiResponse::<Document>::error(
                "MISSING_FILE".to_string(),
                "文件是必需的".to_string()
            )));
        }
    };

    let filename = filename.unwrap_or_else(|| "unknown".to_string());
    let title = request.title.unwrap_or_else(|| filename.clone());

    // 处理文档上传
    match state.rag_service.upload_document(
        title,
        file_data,
        filename,
        request.workspace_id,
        request.tags.unwrap_or_default(),
        request.category,
    ).await {
        Ok(document) => {
            info!("文档上传成功: {}", document.id);
            (StatusCode::CREATED, Json(ApiResponse::success(document)))
        }
        Err(e) => {
            error!("文档上传失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<Document>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 获取文档
async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.rag_service.get_document(id).await {
        Ok(Some(document)) => (StatusCode::OK, Json(ApiResponse::success(document))),
        Ok(None) => (StatusCode::NOT_FOUND, Json(ApiResponse::<Document>::error(
            "DOCUMENT_NOT_FOUND".to_string(),
            format!("文档 {} 不存在", id)
        ))),
        Err(e) => {
            error!("获取文档失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<Document>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 更新文档
async fn update_document(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(update_req): Json<serde_json::Value>,
) -> impl IntoResponse {
    match state.rag_service.update_document(id, update_req).await {
        Ok(document) => (StatusCode::OK, Json(ApiResponse::success(document))),
        Err(e) => {
            error!("更新文档失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<Document>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 删除文档
async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.rag_service.delete_document(id).await {
        Ok(_) => (StatusCode::NO_CONTENT, Json(ApiResponse::<()>::success(()))),
        Err(e) => {
            error!("删除文档失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<()>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 获取文档块
async fn get_document_chunks(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.rag_service.get_document_chunks(id).await {
        Ok(chunks) => (StatusCode::OK, Json(ApiResponse::success(chunks))),
        Err(e) => {
            error!("获取文档块失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<Vec<Chunk>>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 重新索引文档
async fn reindex_document(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.rag_service.reindex_document(id).await {
        Ok(_) => (StatusCode::ACCEPTED, Json(ApiResponse::<()>::success(()))),
        Err(e) => {
            error!("重新索引文档失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<()>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 批量上传文档
async fn batch_upload_documents(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // 批量上传的简化实现
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error(
        "NOT_IMPLEMENTED".to_string(),
        "批量上传功能尚未实现".to_string()
    )))
}

// ============================================================================
// 检索功能处理器
// ============================================================================

/// 搜索请求
#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<u32>,
    strategy: Option<String>,
    filters: Option<Vec<Filter>>,
    workspace_id: Option<Uuid>,
    similarity_threshold: Option<f32>,
}

/// 搜索文档
async fn search_documents(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let query = Query {
        id: Uuid::new_v4(),
        text: req.query,
        options: QueryOptions {
            strategy: req.strategy.unwrap_or_else(|| state.config.retrieval.default_strategy.clone()),
            top_k: req.top_k.unwrap_or(state.config.retrieval.default_top_k),
            similarity_threshold: req.similarity_threshold,
            filters: req.filters.unwrap_or_default(),
            enable_reranking: state.config.retrieval.reranking.enabled,
            rerank_top_k: Some(state.config.retrieval.reranking.top_k),
            workspace_id: req.workspace_id,
        },
        timestamp: chrono::Utc::now(),
    };

    match state.rag_service.search(query).await {
        Ok(results) => (StatusCode::OK, Json(ApiResponse::success(results))),
        Err(e) => {
            error!("搜索失败: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::<SearchResult>::error(
                e.error_code().to_string(),
                e.to_string()
            )))
        }
    }
}

/// 搜索建议
async fn search_suggestions(
    State(_state): State<AppState>,
    Json(_req): Json<serde_json::Value>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error(
        "NOT_IMPLEMENTED".to_string(),
        "搜索建议功能尚未实现".to_string()
    )))
}

/// 查找相似文档
async fn find_similar(
    State(_state): State<AppState>,
    Json(_req): Json<serde_json::Value>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error(
        "NOT_IMPLEMENTED".to_string(),
        "查找相似文档功能尚未实现".to_string()
    )))
}

/// 混合搜索
async fn hybrid_search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    // 强制使用混合策略
    let mut req = req;
    req.strategy = Some("hybrid".to_string());

    search_documents(State(state), Json(req)).await
}

// ============================================================================
// 其他处理器的占位实现
// ============================================================================

// 对话功能处理器
async fn chat_completion(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn chat_stream(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn list_conversations(State(_): State<AppState>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn create_conversation(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn get_conversation(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn delete_conversation(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn get_conversation_messages(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

// 嵌入功能处理器
async fn generate_embeddings(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn batch_generate_embeddings(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn list_embedding_models(State(_): State<AppState>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

// 工作空间管理处理器
async fn list_workspaces(State(_): State<AppState>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn create_workspace(State(_): State<AppState>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn get_workspace(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn update_workspace(State(_): State<AppState>, Path(_): Path<Uuid>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn delete_workspace(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn get_workspace_members(State(_): State<AppState>, Path(_): Path<Uuid>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".到_string())))
}

async fn add_workspace_member(State(_): State<AppState>, Path(_): Path<Uuid>, Json(_): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}

async fn remove_workspace_member(State(_): State<AppState>, Path(_): Path<(Uuid, Uuid)>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(ApiResponse::<()>::error("NOT_IMPLEMENTED".to_string(), "功能尚未实现".to_string())))
}