//! # 核心类型定义模块
//!
//! 定义了 RAG 引擎中使用的所有核心数据结构和类型。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::config::{LlmConfig, RetrievalConfig};

/// 文档 ID 类型
pub type DocumentId = Uuid;

/// 块 ID 类型
pub type ChunkId = Uuid;

/// 对话 ID 类型
pub type ConversationId = Uuid;

/// 消息 ID 类型
pub type MessageId = Uuid;

/// 用户 ID 类型
pub type UserId = Uuid;

/// 工作空间 ID 类型
pub type WorkspaceId = Uuid;

/// 嵌入向量类型
pub type EmbeddingVector = Vec<f32>;

/// 元数据类型
pub type Metadata = HashMap<String, serde_json::Value>;

/// 文档结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// 文档 ID
    pub id: DocumentId,
    /// 文档标题
    pub title: String,
    /// 文档内容
    pub content: String,
    /// 文档元数据
    pub metadata: DocumentMetadata,
    /// 文档块列表
    pub chunks: Vec<Chunk>,
    /// 版本号
    pub version: u64,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 文档状态
    pub status: DocumentStatus,
}

/// 文档元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// 文件名
    pub filename: Option<String>,
    /// 文件类型
    pub file_type: Option<String>,
    /// 文件大小 (字节)
    pub file_size: Option<u64>,
    /// MIME 类型
    pub mime_type: Option<String>,
    /// 源 URL
    pub source_url: Option<String>,
    /// 作者
    pub author: Option<String>,
    /// 语言
    pub language: Option<String>,
    /// 标签
    pub tags: Vec<String>,
    /// 分类
    pub category: Option<String>,
    /// 自定义元数据
    pub custom: Metadata,
}

/// 文档状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DocumentStatus {
    /// 处理中
    Processing,
    /// 已完成
    Completed,
    /// 处理失败
    Failed,
    /// 已删除
    Deleted,
}

/// 文档块结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// 块 ID
    pub id: ChunkId,
    /// 所属文档 ID
    pub document_id: DocumentId,
    /// 块内容
    pub content: String,
    /// 块元数据
    pub metadata: ChunkMetadata,
    /// 嵌入向量
    pub embedding: Option<EmbeddingVector>,
    /// 在文档中的位置
    pub position: ChunkPosition,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

/// 块元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// 块类型 (text, table, image, etc.)
    pub chunk_type: String,
    /// 字符数
    pub char_count: usize,
    /// 词数
    pub word_count: usize,
    /// 语言
    pub language: Option<String>,
    /// 重要性分数
    pub importance_score: Option<f32>,
    /// 自定义元数据
    pub custom: Metadata,
}

/// 块在文档中的位置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkPosition {
    /// 起始字符位置
    pub start_char: usize,
    /// 结束字符位置
    pub end_char: usize,
    /// 页码 (如果适用)
    pub page_number: Option<u32>,
    /// 行号 (如果适用)
    pub line_number: Option<u32>,
}

/// 查询结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// 查询 ID
    pub id: Uuid,
    /// 查询文本
    pub text: String,
    /// 查询选项
    pub options: QueryOptions,
    /// 查询时间
    pub timestamp: DateTime<Utc>,
}

/// 查询选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    /// 检索策略
    pub strategy: String,
    /// 返回结果数量
    pub top_k: u32,
    /// 相似度阈值
    pub similarity_threshold: Option<f32>,
    /// 过滤条件
    pub filters: Vec<Filter>,
    /// 是否启用重排序
    pub enable_reranking: bool,
    /// 重排序数量
    pub rerank_top_k: Option<u32>,
    /// 工作空间 ID
    pub workspace_id: Option<WorkspaceId>,
}

/// 过滤条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    /// 字段名
    pub field: String,
    /// 操作符
    pub operator: FilterOperator,
    /// 值
    pub value: serde_json::Value,
}

/// 过滤操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    /// 等于
    Eq,
    /// 不等于
    Ne,
    /// 大于
    Gt,
    /// 大于等于
    Gte,
    /// 小于
    Lt,
    /// 小于等于
    Lte,
    /// 包含 (数组/字符串)
    Contains,
    /// 不包含
    NotContains,
    /// 在范围内
    In,
    /// 不在范围内
    NotIn,
}

/// 搜索结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// 查询 ID
    pub query_id: Uuid,
    /// 搜索结果项
    pub results: Vec<SearchResultItem>,
    /// 搜索元数据
    pub metadata: SearchMetadata,
    /// 搜索时间
    pub timestamp: DateTime<Utc>,
}

/// 搜索结果项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultItem {
    /// 文档块
    pub chunk: Chunk,
    /// 相似度分数
    pub score: f32,
    /// 排名
    pub rank: u32,
    /// 高亮片段
    pub highlights: Vec<String>,
    /// 相关性解释
    pub explanation: Option<String>,
}

/// 搜索元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetadata {
    /// 总的搜索时间 (毫秒)
    pub total_time_ms: u64,
    /// 检索时间 (毫秒)
    pub retrieval_time_ms: u64,
    /// 重排序时间 (毫秒)
    pub reranking_time_ms: Option<u64>,
    /// 使用的策略
    pub strategy_used: String,
    /// 总的候选数量
    pub total_candidates: u32,
    /// 过滤后的数量
    pub filtered_count: u32,
    /// 最终返回数量
    pub returned_count: u32,
    /// 索引统计
    pub index_stats: IndexStats,
}

/// 索引统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// 总文档数
    pub total_documents: u64,
    /// 总块数
    pub total_chunks: u64,
    /// 索引大小 (字节)
    pub index_size_bytes: u64,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
}

/// 对话结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// 对话 ID
    pub id: ConversationId,
    /// 用户 ID
    pub user_id: UserId,
    /// 工作空间 ID
    pub workspace_id: Option<WorkspaceId>,
    /// 对话标题
    pub title: String,
    /// 消息列表
    pub messages: Vec<Message>,
    /// 对话配置
    pub config: ConversationConfig,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 对话状态
    pub status: ConversationStatus,
}

/// 对话配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// LLM 提供商
    pub llm_provider: String,
    /// 模型名称
    pub model_name: String,
    /// 温度参数
    pub temperature: f32,
    /// 最大 token 数
    pub max_tokens: u32,
    /// 系统提示
    pub system_prompt: Option<String>,
    /// 是否启用 RAG
    pub enable_rag: bool,
    /// RAG 配置
    pub rag_config: Option<RagChatConfig>,
}

/// RAG 聊天配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChatConfig {
    /// 检索数量
    pub retrieval_top_k: u32,
    /// 相似度阈值
    pub similarity_threshold: f32,
    /// 是否显示来源
    pub show_sources: bool,
    /// 上下文长度限制
    pub context_length_limit: u32,
}

/// 对话状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConversationStatus {
    /// 活跃
    Active,
    /// 已归档
    Archived,
    /// 已删除
    Deleted,
}

/// 消息结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 消息 ID
    pub id: MessageId,
    /// 对话 ID
    pub conversation_id: ConversationId,
    /// 消息类型
    pub message_type: MessageType,
    /// 消息内容
    pub content: String,
    /// 消息元数据
    pub metadata: MessageMetadata,
    /// 创建时间
    pub timestamp: DateTime<Utc>,
    /// 父消息 ID (用于分支对话)
    pub parent_id: Option<MessageId>,
}

/// 消息类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    /// 用户消息
    User,
    /// 助手消息
    Assistant,
    /// 系统消息
    System,
    /// 工具调用消息
    ToolCall,
    /// 工具响应消息
    ToolResponse,
}

/// 消息元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// 使用的模型
    pub model_used: Option<String>,
    /// 生成时间 (毫秒)
    pub generation_time_ms: Option<u64>,
    /// 使用的 token 数
    pub token_count: Option<u32>,
    /// 相关的源文档
    pub sources: Vec<MessageSource>,
    /// 生成参数
    pub generation_params: Option<GenerationParams>,
    /// 自定义元数据
    pub custom: Metadata,
}

/// 消息来源
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSource {
    /// 文档 ID
    pub document_id: DocumentId,
    /// 块 ID
    pub chunk_id: ChunkId,
    /// 相似度分数
    pub score: f32,
    /// 引用文本
    pub excerpt: String,
}

/// 生成参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// 温度
    pub temperature: f32,
    /// top_p
    pub top_p: Option<f32>,
    /// top_k
    pub top_k: Option<u32>,
    /// 频率惩罚
    pub frequency_penalty: Option<f32>,
    /// 存在惩罚
    pub presence_penalty: Option<f32>,
    /// 停止词
    pub stop_sequences: Vec<String>,
}

/// 用户结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// 用户 ID
    pub id: UserId,
    /// 用户名
    pub username: String,
    /// 邮箱
    pub email: String,
    /// 显示名称
    pub display_name: String,
    /// 用户角色
    pub role: UserRole,
    /// 用户设置
    pub settings: UserSettings,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 最后登录时间
    pub last_login: Option<DateTime<Utc>>,
    /// 是否激活
    pub is_active: bool,
}

/// 用户角色
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserRole {
    /// 管理员
    Admin,
    /// 编辑者
    Editor,
    /// 查看者
    Viewer,
    /// 客人
    Guest,
}

/// 用户设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    /// 语言偏好
    pub language: String,
    /// 时区
    pub timezone: String,
    /// 主题
    pub theme: String,
    /// 通知设置
    pub notifications: NotificationSettings,
    /// API 偏好
    pub api_preferences: ApiPreferences,
}

/// 通知设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// 邮件通知
    pub email_notifications: bool,
    /// 推送通知
    pub push_notifications: bool,
    /// 系统通知
    pub system_notifications: bool,
}

/// API 偏好设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPreferences {
    /// 默认 LLM 提供商
    pub default_llm_provider: String,
    /// 默认嵌入提供商
    pub default_embedding_provider: String,
    /// 默认检索策略
    pub default_retrieval_strategy: String,
    /// 默认 top-k
    pub default_top_k: u32,
}

/// 工作空间结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    /// 工作空间 ID
    pub id: WorkspaceId,
    /// 工作空间名称
    pub name: String,
    /// 描述
    pub description: Option<String>,
    /// 所有者 ID
    pub owner_id: UserId,
    /// 成员列表
    pub members: Vec<WorkspaceMember>,
    /// 工作空间设置
    pub settings: WorkspaceSettings,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 是否激活
    pub is_active: bool,
}

/// 工作空间成员
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMember {
    /// 用户 ID
    pub user_id: UserId,
    /// 成员角色
    pub role: WorkspaceRole,
    /// 加入时间
    pub joined_at: DateTime<Utc>,
    /// 权限列表
    pub permissions: Vec<Permission>,
}

/// 工作空间角色
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkspaceRole {
    /// 所有者
    Owner,
    /// 管理员
    Admin,
    /// 编辑者
    Editor,
    /// 查看者
    Viewer,
}

/// 权限枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Permission {
    /// 读取文档
    ReadDocuments,
    /// 写入文档
    WriteDocuments,
    /// 删除文档
    DeleteDocuments,
    /// 管理成员
    ManageMembers,
    /// 管理设置
    ManageSettings,
    /// 查看分析
    ViewAnalytics,
}

/// 工作空间设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSettings {
    /// 默认检索配置
    pub default_retrieval_config: RetrievalConfig,
    /// 默认 LLM 配置
    pub default_llm_config: LlmConfig,
    /// 文档保留策略
    pub document_retention_days: Option<u32>,
    /// 最大文档数量
    pub max_documents: Option<u64>,
    /// 最大存储空间 (字节)
    pub max_storage_bytes: Option<u64>,
    /// 自定义设置
    pub custom_settings: Metadata,
}

/// 批处理任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTask {
    /// 任务 ID
    pub id: Uuid,
    /// 任务类型
    pub task_type: BatchTaskType,
    /// 任务状态
    pub status: BatchTaskStatus,
    /// 任务参数
    pub parameters: Metadata,
    /// 任务结果
    pub result: Option<BatchTaskResult>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 开始时间
    pub started_at: Option<DateTime<Utc>>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
    /// 错误信息
    pub error: Option<String>,
    /// 进度信息
    pub progress: TaskProgress,
}

/// 批处理任务类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BatchTaskType {
    /// 文档索引
    DocumentIndexing,
    /// 批量嵌入
    BatchEmbedding,
    /// 索引重建
    IndexRebuild,
    /// 数据导入
    DataImport,
    /// 数据导出
    DataExport,
    /// 清理任务
    Cleanup,
}

/// 批处理任务状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BatchTaskStatus {
    /// 等待中
    Pending,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
}

/// 任务进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    /// 已完成项目数
    pub completed_items: u64,
    /// 总项目数
    pub total_items: u64,
    /// 进度百分比 (0-100)
    pub percentage: f32,
    /// 当前阶段
    pub current_stage: String,
    /// 预估剩余时间 (秒)
    pub estimated_remaining_seconds: Option<u64>,
}

/// 批处理任务结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTaskResult {
    /// 成功处理的项目数
    pub successful_items: u64,
    /// 失败的项目数
    pub failed_items: u64,
    /// 跳过的项目数
    pub skipped_items: u64,
    /// 处理详情
    pub details: Metadata,
}

/// 系统健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// 整体状态
    pub status: HealthStatusLevel,
    /// 检查时间
    pub timestamp: DateTime<Utc>,
    /// 系统信息
    pub system_info: SystemInfo,
    /// 组件状态
    pub components: Vec<ComponentHealth>,
    /// 性能指标
    pub metrics: PerformanceMetrics,
}

/// 健康状态级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatusLevel {
    /// 健康
    Healthy,
    /// 警告
    Warning,
    /// 关键
    Critical,
    /// 未知
    Unknown,
}

/// 系统信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// 版本
    pub version: String,
    /// 启动时间
    pub uptime_seconds: u64,
    /// 运行环境
    pub environment: String,
    /// 主机名
    pub hostname: String,
    /// CPU 核心数
    pub cpu_cores: u32,
    /// 总内存 (字节)
    pub total_memory_bytes: u64,
}

/// 组件健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// 组件名称
    pub name: String,
    /// 状态
    pub status: HealthStatusLevel,
    /// 响应时间 (毫秒)
    pub response_time_ms: Option<u64>,
    /// 错误信息
    pub error: Option<String>,
    /// 最后检查时间
    pub last_check: DateTime<Utc>,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU 使用率 (0-1)
    pub cpu_usage: f32,
    /// 内存使用率 (0-1)
    pub memory_usage: f32,
    /// 磁盘使用率 (0-1)
    pub disk_usage: f32,
    /// 活跃连接数
    pub active_connections: u64,
    /// 请求速率 (每秒)
    pub requests_per_second: f32,
    /// 平均响应时间 (毫秒)
    pub avg_response_time_ms: f32,
    /// 错误率 (0-1)
    pub error_rate: f32,
}

/// API 响应包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// 是否成功
    pub success: bool,
    /// 响应数据
    pub data: Option<T>,
    /// 错误信息
    pub error: Option<ApiError>,
    /// 元数据
    pub metadata: Option<ResponseMetadata>,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
}

/// API 错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// 错误码
    pub code: String,
    /// 错误消息
    pub message: String,
    /// 详细信息
    pub details: Option<Metadata>,
    /// 请求 ID
    pub request_id: Option<String>,
}

/// 响应元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// 请求 ID
    pub request_id: String,
    /// 处理时间 (毫秒)
    pub processing_time_ms: u64,
    /// API 版本
    pub api_version: String,
    /// 分页信息
    pub pagination: Option<PaginationInfo>,
}

/// 分页信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationInfo {
    /// 当前页
    pub page: u32,
    /// 每页大小
    pub page_size: u32,
    /// 总页数
    pub total_pages: u32,
    /// 总项目数
    pub total_items: u64,
    /// 是否有下一页
    pub has_next: bool,
    /// 是否有上一页
    pub has_previous: bool,
}

/// 事件结构体 (用于审计和日志)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// 事件 ID
    pub id: Uuid,
    /// 事件类型
    pub event_type: String,
    /// 用户 ID
    pub user_id: Option<UserId>,
    /// 工作空间 ID
    pub workspace_id: Option<WorkspaceId>,
    /// 资源 ID
    pub resource_id: Option<String>,
    /// 资源类型
    pub resource_type: Option<String>,
    /// 操作
    pub action: String,
    /// 事件数据
    pub data: Metadata,
    /// IP 地址
    pub ip_address: Option<String>,
    /// 用户代理
    pub user_agent: Option<String>,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
}

/// 实用函数实现
impl Document {
    /// 创建新文档
    pub fn new(title: String, content: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title,
            content,
            metadata: DocumentMetadata::default(),
            chunks: Vec::new(),
            version: 1,
            created_at: now,
            updated_at: now,
            status: DocumentStatus::Processing,
        }
    }

    /// 获取文档总字符数
    pub fn char_count(&self) -> usize {
        self.content.len()
    }

    /// 获取文档总词数
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// 检查文档是否已完成处理
    pub fn is_completed(&self) -> bool {
        self.status == DocumentStatus::Completed
    }
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            filename: None,
            file_type: None,
            file_size: None,
            mime_type: None,
            source_url: None,
            author: None,
            language: None,
            tags: Vec::new(),
            category: None,
            custom: HashMap::new(),
        }
    }
}

impl Chunk {
    /// 创建新文档块
    pub fn new(document_id: DocumentId, content: String, position: ChunkPosition) -> Self {
        Self {
            id: Uuid::new_v4(),
            document_id,
            content: content.clone(),
            metadata: ChunkMetadata::from_content(&content),
            embedding: None,
            position,
            created_at: Utc::now(),
        }
    }

    /// 检查是否有嵌入向量
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

impl ChunkMetadata {
    /// 从内容创建块元数据
    pub fn from_content(content: &str) -> Self {
        Self {
            chunk_type: "text".to_string(),
            char_count: content.len(),
            word_count: content.split_whitespace().count(),
            language: None,
            importance_score: None,
            custom: HashMap::new(),
        }
    }
}

impl Query {
    /// 创建新查询
    pub fn new(text: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            options: QueryOptions::default(),
            timestamp: Utc::now(),
        }
    }

    /// 使用选项创建查询
    pub fn with_options(text: String, options: QueryOptions) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            options,
            timestamp: Utc::now(),
        }
    }
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            strategy: "hybrid".to_string(),
            top_k: 10,
            similarity_threshold: None,
            filters: Vec::new(),
            enable_reranking: false,
            rerank_top_k: None,
            workspace_id: None,
        }
    }
}

impl<T> ApiResponse<T> {
    /// 创建成功响应
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            metadata: None,
            timestamp: Utc::now(),
        }
    }

    /// 创建错误响应
    pub fn error(code: String, message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(ApiError {
                code,
                message,
                details: None,
                request_id: None,
            }),
            metadata: None,
            timestamp: Utc::now(),
        }
    }

    /// 添加元数据
    pub fn with_metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl TaskProgress {
    /// 创建新的任务进度
    pub fn new(total_items: u64) -> Self {
        Self {
            completed_items: 0,
            total_items,
            percentage: 0.0,
            current_stage: "Starting".to_string(),
            estimated_remaining_seconds: None,
        }
    }

    /// 更新进度
    pub fn update(&mut self, completed_items: u64, current_stage: String) {
        self.completed_items = completed_items;
        self.percentage = if self.total_items > 0 {
            (completed_items as f32 / self.total_items as f32) * 100.0
        } else {
            0.0
        };
        self.current_stage = current_stage;
    }

    /// 检查是否完成
    pub fn is_completed(&self) -> bool {
        self.completed_items >= self.total_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Test Document".to_string(), "Test content".to_string());
        assert_eq!(doc.title, "Test Document");
        assert_eq!(doc.content, "Test content");
        assert_eq!(doc.version, 1);
        assert_eq!(doc.status, DocumentStatus::Processing);
        assert_eq!(doc.char_count(), 12);
        assert_eq!(doc.word_count(), 2);
    }

    #[test]
    fn test_chunk_creation() {
        let doc_id = Uuid::new_v4();
        let position = ChunkPosition {
            start_char: 0,
            end_char: 10,
            page_number: Some(1),
            line_number: Some(1),
        };
        let chunk = Chunk::new(doc_id, "Test chunk".to_string(), position);

        assert_eq!(chunk.document_id, doc_id);
        assert_eq!(chunk.content, "Test chunk");
        assert!(!chunk.has_embedding());
        assert_eq!(chunk.metadata.char_count, 10);
        assert_eq!(chunk.metadata.word_count, 2);
    }

    #[test]
    fn test_api_response() {
        let response = ApiResponse::success("test data".to_string());
        assert!(response.success);
        assert_eq!(response.data.unwrap(), "test data");
        assert!(response.error.is_none());

        let error_response = ApiResponse::<String>::error(
            "TEST_ERROR".to_string(),
            "Test error message".to_string(),
        );
        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert!(error_response.error.is_some());
    }

    #[test]
    fn test_task_progress() {
        let mut progress = TaskProgress::new(100);
        assert_eq!(progress.percentage, 0.0);
        assert!(!progress.is_completed());

        progress.update(50, "Processing".to_string());
        assert_eq!(progress.percentage, 50.0);
        assert_eq!(progress.current_stage, "Processing");
        assert!(!progress.is_completed());

        progress.update(100, "Completed".to_string());
        assert_eq!(progress.percentage, 100.0);
        assert!(progress.is_completed());
    }
}