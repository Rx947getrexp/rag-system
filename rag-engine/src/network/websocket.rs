//! # WebSocket 服务器模块
//!
//! 提供实时通信功能，支持流式搜索和对话

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::Response,
    routing::get,
    Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use tokio::{
    net::TcpListener,
    sync::{broadcast, RwLock},
    time::interval,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    config::RagConfig,
    error::{RagError, RagResult},
    services::rag_service::RagService,
    types::*,
};

/// WebSocket 消息类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// 搜索请求
    Search {
        id: String,
        query: String,
        options: SearchOptions,
    },
    /// 搜索结果
    SearchResult {
        id: String,
        results: Vec<SearchResultItem>,
        is_final: bool,
    },
    /// 对话请求
    Chat {
        id: String,
        message: String,
        conversation_id: Option<String>,
    },
    /// 对话响应 (流式)
    ChatResponse {
        id: String,
        content: String,
        is_final: bool,
        conversation_id: String,
    },
    /// 错误消息
    Error {
        id: Option<String>,
        code: String,
        message: String,
    },
    /// 心跳
    Ping,
    /// 心跳响应
    Pong,
    /// 连接确认
    Connected {
        session_id: String,
    },
}

/// 搜索选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    pub top_k: Option<u32>,
    pub strategy: Option<String>,
    pub workspace_id: Option<String>,
}

/// WebSocket 连接信息
#[derive(Debug, Clone)]
struct WsConnection {
    id: Uuid,
    user_id: Option<Uuid>,
    workspace_id: Option<Uuid>,
    connected_at: chrono::DateTime<chrono::Utc>,
    last_ping: chrono::DateTime<chrono::Utc>,
}

/// WebSocket 服务器状态
#[derive(Clone)]
pub struct WsState {
    pub config: Arc<RagConfig>,
    pub rag_service: Arc<RagService>,
    pub connections: Arc<RwLock<HashMap<Uuid, WsConnection>>>,
    pub broadcast_tx: broadcast::Sender<WsMessage>,
}

/// WebSocket 服务器
pub struct WebSocketServer {
    config: Arc<RagConfig>,
    rag_service: Arc<RagService>,
}

impl WebSocketServer {
    /// 创建新的 WebSocket 服务器
    pub async fn new(
        config: Arc<RagConfig>,
        rag_service: Arc<RagService>,
    ) -> RagResult<Self> {
        Ok(Self {
            config,
            rag_service,
        })
    }

    /// 启动 WebSocket 服务器
    pub async fn serve(self) -> RagResult<()> {
        let addr: SocketAddr = self.config.network.websocket.bind_address
            .parse()
            .map_err(|e| RagError::NetworkError(
                crate::error::NetworkError::ServerBindingFailed {
                    address: self.config.network.websocket.bind_address.clone(),
                }
            ))?;

        info!("🔌 启动 WebSocket 服务器: {}", addr);

        // 创建广播通道
        let (broadcast_tx, _) = broadcast::channel(1000);

        // 创建服务器状态
        let state = WsState {
            config: self.config.clone(),
            rag_service: self.rag_service,
            connections: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx: broadcast_tx.clone(),
        };

        // 启动心跳任务
        let heartbeat_state = state.clone();
        tokio::spawn(async move {
            heartbeat_task(heartbeat_state).await;
        });

        // 启动连接清理任务
        let cleanup_state = state.clone();
        tokio::spawn(async move {
            connection_cleanup_task(cleanup_state).await;
        });

        // 构建路由
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .route("/ws/health", get(ws_health))
            .route("/ws/stats", get(ws_stats))
            .with_state(state);

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
}

/// WebSocket 连接处理器
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// 处理 WebSocket 连接
async fn handle_socket(socket: WebSocket, state: WsState) {
    let connection_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // 注册连接
    {
        let mut connections = state.connections.write().await;
        connections.insert(connection_id, WsConnection {
            id: connection_id,
            user_id: None,
            workspace_id: None,
            connected_at: now,
            last_ping: now,
        });
    }

    info!("WebSocket 连接建立: {}", connection_id);

    let (mut sender, mut receiver) = socket.split();

    // 发送连接确认
    let connected_msg = WsMessage::Connected {
        session_id: connection_id.to_string(),
    };

    if let Ok(msg_text) = serde_json::to_string(&connected_msg) {
        let _ = sender.send(Message::Text(msg_text)).await;
    }

    // 创建广播接收器
    let mut broadcast_rx = state.broadcast_tx.subscribe();

    // 处理广播消息的任务
    let broadcast_sender = sender.clone();
    let broadcast_task = tokio::spawn(async move {
        while let Ok(msg) = broadcast_rx.recv().await {
            if let Ok(msg_text) = serde_json::to_string(&msg) {
                if broadcast_sender.send(Message::Text(msg_text)).await.is_err() {
                    break;
                }
            }
        }
    });

    // 处理接收到的消息
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Err(e) = handle_ws_message(text, &mut sender, &state, connection_id).await {
                    error!("处理 WebSocket 消息失败: {}", e);
                    let error_msg = WsMessage::Error {
                        id: None,
                        code: "INTERNAL_ERROR".to_string(),
                        message: e.to_string(),
                    };

                    if let Ok(error_text) = serde_json::to_string(&error_msg) {
                        let _ = sender.send(Message::Text(error_text)).await;
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                warn!("收到二进制消息，暂不支持");
            }
            Ok(Message::Ping(data)) => {
                // 响应 ping
                let _ = sender.send(Message::Pong(data)).await;

                // 更新最后 ping 时间
                {
                    let mut connections = state.connections.write().await;
                    if let Some(conn) = connections.get_mut(&connection_id) {
                        conn.last_ping = chrono::Utc::now();
                    }
                }
            }
            Ok(Message::Pong(_)) => {
                // 更新最后 ping 时间
                {
                    let mut connections = state.connections.write().await;
                    if let Some(conn) = connections.get_mut(&connection_id) {
                        conn.last_ping = chrono::Utc::now();
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket 连接关闭: {}", connection_id);
                break;
            }
            Err(e) => {
                error!("WebSocket 错误: {}", e);
                break;
            }
        }
    }

    // 清理连接
    {
        let mut connections = state.connections.write().await;
        connections.remove(&connection_id);
    }

    // 取消广播任务
    broadcast_task.abort();

    info!("WebSocket 连接清理完成: {}", connection_id);
}

/// 处理 WebSocket 消息
async fn handle_ws_message(
    text: String,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    state: &WsState,
    connection_id: Uuid,
) -> RagResult<()> {
    let ws_msg: WsMessage = serde_json::from_str(&text)
        .map_err(|e| RagError::ValidationError(format!("无效的消息格式: {}", e)))?;

    debug!("收到 WebSocket 消息: {:?}", ws_msg);

    match ws_msg {
        WsMessage::Search { id, query, options } => {
            handle_search_request(id, query, options, sender, state).await?;
        }
        WsMessage::Chat { id, message, conversation_id } => {
            handle_chat_request(id, message, conversation_id, sender, state).await?;
        }
        WsMessage::Ping => {
            let pong = WsMessage::Pong;
            if let Ok(pong_text) = serde_json::to_string(&pong) {
                sender.send(Message::Text(pong_text)).await
                    .map_err(|e| RagError::NetworkError(
                        crate::error::NetworkError::WebSocketConnectionFailed(e.to_string())
                    ))?;
            }
        }
        _ => {
            warn!("收到未处理的消息类型");
        }
    }

    Ok(())
}

/// 处理搜索请求
async fn handle_search_request(
    request_id: String,
    query_text: String,
    options: SearchOptions,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    state: &WsState,
) -> RagResult<()> {
    // 构建查询
    let query = Query {
        id: Uuid::new_v4(),
        text: query_text,
        options: QueryOptions {
            strategy: options.strategy.unwrap_or_else(|| "hybrid".to_string()),
            top_k: options.top_k.unwrap_or(10),
            similarity_threshold: None,
            filters: Vec::new(),
            enable_reranking: false,
            rerank_top_k: None,
            workspace_id: options.workspace_id.and_then(|id| id.parse().ok()),
        },
        timestamp: chrono::Utc::now(),
    };

    // 执行搜索
    match state.rag_service.search(query).await {
        Ok(search_result) => {
            let response = WsMessage::SearchResult {
                id: request_id,
                results: search_result.results,
                is_final: true,
            };

            if let Ok(response_text) = serde_json::to_string(&response) {
                sender.send(Message::Text(response_text)).await
                    .map_err(|e| RagError::NetworkError(
                        crate::error::NetworkError::WebSocketConnectionFailed(e.to_string())
                    ))?;
            }
        }
        Err(e) => {
            let error_msg = WsMessage::Error {
                id: Some(request_id),
                code: e.error_code().to_string(),
                message: e.to_string(),
            };

            if let Ok(error_text) = serde_json::to_string(&error_msg) {
                sender.send(Message::Text(error_text)).await
                    .map_err(|e| RagError::NetworkError(
                        crate::error::NetworkError::WebSocketConnectionFailed(e.to_string())
                    ))?;
            }
        }
    }

    Ok(())
}

/// 处理对话请求
async fn handle_chat_request(
    request_id: String,
    _message: String,
    _conversation_id: Option<String>,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    _state: &WsState,
) -> RagResult<()> {
    // 简化实现 - 返回未实现错误
    let error_msg = WsMessage::Error {
        id: Some(request_id),
        code: "NOT_IMPLEMENTED".to_string(),
        message: "对话功能尚未实现".to_string(),
    };

    if let Ok(error_text) = serde_json::to_string(&error_msg) {
        sender.send(Message::Text(error_text)).await
            .map_err(|e| RagError::NetworkError(
                crate::error::NetworkError::WebSocketConnectionFailed(e.to_string())
            ))?;
    }

    Ok(())
}

/// WebSocket 健康检查
async fn ws_health(State(state): State<WsState>) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let connections_count = state.connections.read().await.len();

    Ok(axum::Json(serde_json::json!({
        "status": "healthy",
        "active_connections": connections_count,
        "timestamp": chrono::Utc::now()
    })))
}

/// WebSocket 统计信息
async fn ws_stats(State(state): State<WsState>) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let connections = state.connections.read().await;
    let connections_count = connections.len();
    let avg_connection_time = if connections_count > 0 {
        let now = chrono::Utc::now();
        let total_duration: i64 = connections.values()
            .map(|conn| (now - conn.connected_at).num_seconds())
            .sum();
        total_duration / connections_count as i64
    } else {
        0
    };

    Ok(axum::Json(serde_json::json!({
        "active_connections": connections_count,
        "average_connection_duration_seconds": avg_connection_time,
        "max_connections": state.config.network.websocket.max_connections,
        "timestamp": chrono::Utc::now()
    })))
}

/// 心跳任务
async fn heartbeat_task(state: WsState) {
    let mut interval = interval(Duration::from_secs(
        state.config.network.websocket.heartbeat_interval
    ));

    loop {
        interval.tick().await;

        let ping_msg = WsMessage::Ping;
        if let Ok(_) = serde_json::to_string(&ping_msg) {
            // 广播心跳消息
            let _ = state.broadcast_tx.send(ping_msg);
        }
    }
}

/// 连接清理任务
async fn connection_cleanup_task(state: WsState) {
    let mut interval = interval(Duration::from_secs(60)); // 每分钟清理一次

    loop {
        interval.tick().await;

        let now = chrono::Utc::now();
        let timeout_duration = chrono::Duration::seconds(
            (state.config.network.websocket.heartbeat_interval * 3) as i64
        );

        {
            let mut connections = state.connections.write().await;
            let mut to_remove = Vec::new();

            for (id, conn) in connections.iter() {
                if now - conn.last_ping > timeout_duration {
                    to_remove.push(*id);
                }
            }

            for id in to_remove {
                connections.remove(&id);
                info!("清理超时的 WebSocket 连接: {}", id);
            }
        }
    }
}