//! # LLM 集成模块
//!
//! 支持多种大语言模型，包括 OpenAI、Anthropic、本地模型等
//! 文件路径: rag-engine/src/llm/mod.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::error::{RagError, RagResult};
use crate::types::SearchResultItem;

/// LLM 提供商
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    HuggingFace,
    Ollama,
    Local,
}

/// LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub model_name: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub timeout_seconds: u32,
    pub system_prompt: Option<String>,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// 消息角色
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Function,
}

/// 聊天请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub id: Uuid,
    pub messages: Vec<ChatMessage>,
    pub context: Option<Vec<SearchResultItem>>,
    pub stream: bool,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// 聊天响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: Uuid,
    pub content: String,
    pub role: MessageRole,
    pub finish_reason: Option<String>,
    pub usage: TokenUsage,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// 流式响应块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: Uuid,
    pub content: String,
    pub is_final: bool,
    pub finish_reason: Option<String>,
}

/// Token 使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// LLM 服务 trait
#[async_trait]
pub trait LLMService: Send + Sync {
    /// 生成聊天响应
    async fn chat(&self, request: ChatRequest) -> RagResult<ChatResponse>;

    /// 流式聊天响应
    async fn chat_stream(&self, request: ChatRequest) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>>;

    /// 生成 RAG 响应
    async fn generate_rag_response(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<ChatResponse>;

    /// 流式 RAG 响应
    async fn generate_rag_response_stream(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>>;

    /// 生成摘要
    async fn summarize(&self, text: &str, max_length: Option<u32>) -> RagResult<String>;

    /// 提取关键词
    async fn extract_keywords(&self, text: &str, count: u32) -> RagResult<Vec<String>>;

    /// 生成查询建议
    async fn suggest_queries(&self, context: &str, count: u32) -> RagResult<Vec<String>>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// OpenAI LLM 服务实现
pub struct OpenAILLMService {
    client: reqwest::Client,
    config: LLMConfig,
}

impl OpenAILLMService {
    pub fn new(config: LLMConfig) -> RagResult<Self> {
        if config.provider != LLMProvider::OpenAI {
            return Err(RagError::ConfigurationError(
                "配置提供商必须是 OpenAI".to_string()
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds as u64))
            .build()
            .map_err(|e| RagError::LLMError(format!("HTTP 客户端创建失败: {}", e)))?;

        Ok(Self { client, config })
    }

    async fn call_openai_api(&self, messages: &[ChatMessage], stream: bool) -> RagResult<serde_json::Value> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| RagError::ConfigurationError("缺少 OpenAI API 密钥".to_string()))?;

        let base_url = self.config.base_url.as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": match msg.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::Function => "function",
                    },
                    "content": msg.content
                })
            })
            .collect();

        let request_body = serde_json::json!({
            "model": self.config.model_name,
            "messages": openai_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "stream": stream
        });

        let response = self.client
            .post(&format!("{}/chat/completions", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RagError::LLMError(format!("API 请求失败: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(RagError::LLMError(format!(
                "OpenAI API 错误: {}", error_text
            )));
        }

        let response_data: serde_json::Value = response.json().await
            .map_err(|e| RagError::LLMError(format!("响应解析失败: {}", e)))?;

        Ok(response_data)
    }

    fn build_rag_prompt(&self, query: &str, context: &[SearchResultItem], history: Option<&[ChatMessage]>) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 系统提示
        let system_prompt = self.config.system_prompt.as_deref().unwrap_or(
            "你是一个有用的AI助手。请基于提供的上下文信息回答用户问题。如果上下文中没有相关信息，请诚实地说明。"
        );

        messages.push(ChatMessage {
            role: MessageRole::System,
            content: system_prompt.to_string(),
            metadata: None,
        });

        // 添加对话历史
        if let Some(history) = history {
            for msg in history {
                if msg.role != MessageRole::System {
                    messages.push(msg.clone());
                }
            }
        }

        // 构建包含上下文的用户消息
        let mut context_text = String::new();
        if !context.is_empty() {
            context_text.push_str("相关上下文信息：\n\n");
            for (i, item) in context.iter().enumerate() {
                context_text.push_str(&format!("{}. {}\n\n", i + 1, item.content));
            }
        }

        let user_message = if context_text.is_empty() {
            query.to_string()
        } else {
            format!("{}\n\n问题：{}", context_text, query)
        };

        messages.push(ChatMessage {
            role: MessageRole::User,
            content: user_message,
            metadata: None,
        });

        messages
    }
}

#[async_trait]
impl LLMService for OpenAILLMService {
    async fn chat(&self, request: ChatRequest) -> RagResult<ChatResponse> {
        let response_data = self.call_openai_api(&request.messages, false).await?;

        let choice = response_data["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .ok_or_else(|| RagError::LLMError("响应中没有选择项".to_string()))?;

        let message = &choice["message"];
        let content = message["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let finish_reason = choice["finish_reason"]
            .as_str()
            .map(|s| s.to_string());

        let usage = if let Some(usage_data) = response_data["usage"].as_object() {
            TokenUsage {
                prompt_tokens: usage_data["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: usage_data["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: usage_data["total_tokens"].as_u64().unwrap_or(0) as u32,
            }
        } else {
            TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }
        };

        Ok(ChatResponse {
            id: request.id,
            content,
            role: MessageRole::Assistant,
            finish_reason,
            usage,
            metadata: request.metadata,
        })
    }

    async fn chat_stream(&self, request: ChatRequest) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>> {
        use tokio_stream::StreamExt;

        // 简化实现：对于流式响应，实际应该处理 Server-Sent Events
        let response = self.chat(request.clone()).await?;

        let chunks = vec![
            StreamChunk {
                id: request.id,
                content: response.content,
                is_final: true,
                finish_reason: response.finish_reason,
            }
        ];

        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Box::new(stream))
    }

    async fn generate_rag_response(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<ChatResponse> {
        let messages = self.build_rag_prompt(query, &context, conversation_history.as_deref());

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: Some(context),
            stream: false,
            metadata: Some(HashMap::from([
                ("query".to_string(), serde_json::Value::String(query.to_string())),
                ("context_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(context.len())
                )),
            ])),
        };

        self.chat(request).await
    }

    async fn generate_rag_response_stream(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>> {
        let messages = self.build_rag_prompt(query, &context, conversation_history.as_deref());

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: Some(context),
            stream: true,
            metadata: Some(HashMap::from([
                ("query".to_string(), serde_json::Value::String(query.to_string())),
            ])),
        };

        self.chat_stream(request).await
    }

    async fn summarize(&self, text: &str, max_length: Option<u32>) -> RagResult<String> {
        let max_length = max_length.unwrap_or(150);

        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("请为以下文本生成一个不超过{}字的摘要。", max_length),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: text.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;
        Ok(response.content)
    }

    async fn extract_keywords(&self, text: &str, count: u32) -> RagResult<Vec<String>> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("请从以下文本中提取{}个最重要的关键词，用逗号分隔。", count),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: text.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;

        // 解析关键词
        let keywords: Vec<String> = response.content
            .split(',')
            .map(|kw| kw.trim().to_string())
            .filter(|kw| !kw.is_empty())
            .collect();

        Ok(keywords)
    }

    async fn suggest_queries(&self, context: &str, count: u32) -> RagResult<Vec<String>> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("基于以下上下文，生成{}个相关的搜索查询建议，每行一个。", count),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: context.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;

        // 解析查询建议
        let suggestions: Vec<String> = response.content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .take(count as usize)
            .collect();

        Ok(suggestions)
    }

    async fn health_check(&self) -> RagResult<()> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::User,
                content: "Hello".to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        self.chat(request).await?;
        Ok(())
    }
}

/// Anthropic Claude 服务实现
pub struct AnthropicLLMService {
    client: reqwest::Client,
    config: LLMConfig,
}

impl AnthropicLLMService {
    pub fn new(config: LLMConfig) -> RagResult<Self> {
        if config.provider != LLMProvider::Anthropic {
            return Err(RagError::ConfigurationError(
                "配置提供商必须是 Anthropic".to_string()
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds as u64))
            .build()
            .map_err(|e| RagError::LLMError(format!("HTTP 客户端创建失败: {}", e)))?;

        Ok(Self { client, config })
    }

    async fn call_anthropic_api(&self, messages: &[ChatMessage]) -> RagResult<serde_json::Value> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| RagError::ConfigurationError("缺少 Anthropic API 密钥".to_string()))?;

        let base_url = self.config.base_url.as_deref()
            .unwrap_or("https://api.anthropic.com/v1");

        // Anthropic API 格式与 OpenAI 略有不同
        let mut system_message = None;
        let mut conversation_messages = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system_message = Some(msg.content.clone());
                },
                _ => {
                    conversation_messages.push(serde_json::json!({
                        "role": match msg.role {
                            MessageRole::User => "user",
                            MessageRole::Assistant => "assistant",
                            _ => "user",
                        },
                        "content": msg.content
                    }));
                }
            }
        }

        let mut request_body = serde_json::json!({
            "model": self.config.model_name,
            "messages": conversation_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        });

        if let Some(system) = system_message {
            request_body["system"] = serde_json::Value::String(system);
        }

        let response = self.client
            .post(&format!("{}/messages", base_url))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RagError::LLMError(format!("API 请求失败: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(RagError::LLMError(format!(
                "Anthropic API 错误: {}", error_text
            )));
        }

        let response_data: serde_json::Value = response.json().await
            .map_err(|e| RagError::LLMError(format!("响应解析失败: {}", e)))?;

        Ok(response_data)
    }
}

#[async_trait]
impl LLMService for AnthropicLLMService {
    async fn chat(&self, request: ChatRequest) -> RagResult<ChatResponse> {
        let response_data = self.call_anthropic_api(&request.messages).await?;

        let content = response_data["content"]
            .as_array()
            .and_then(|content| content.first())
            .and_then(|item| item["text"].as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason = response_data["stop_reason"]
            .as_str()
            .map(|s| s.to_string());

        let usage = if let Some(usage_data) = response_data["usage"].as_object() {
            TokenUsage {
                prompt_tokens: usage_data["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: usage_data["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: (usage_data["input_tokens"].as_u64().unwrap_or(0) +
                    usage_data["output_tokens"].as_u64().unwrap_or(0)) as u32,
            }
        } else {
            TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }
        };

        Ok(ChatResponse {
            id: request.id,
            content,
            role: MessageRole::Assistant,
            finish_reason,
            usage,
            metadata: request.metadata,
        })
    }

    async fn chat_stream(&self, request: ChatRequest) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>> {
        // 简化实现
        let response = self.chat(request.clone()).await?;

        let chunks = vec![
            StreamChunk {
                id: request.id,
                content: response.content,
                is_final: true,
                finish_reason: response.finish_reason,
            }
        ];

        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Box::new(stream))
    }

    async fn generate_rag_response(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<ChatResponse> {
        // 使用与 OpenAI 相同的提示构建逻辑
        let openai_service = OpenAILLMService::new(LLMConfig {
            provider: LLMProvider::OpenAI,
            ..self.config.clone()
        })?;

        let messages = openai_service.build_rag_prompt(query, &context, conversation_history.as_deref());

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: Some(context),
            stream: false,
            metadata: None,
        };

        self.chat(request).await
    }

    async fn generate_rag_response_stream(
        &self,
        query: &str,
        context: Vec<SearchResultItem>,
        conversation_history: Option<Vec<ChatMessage>>,
    ) -> RagResult<Box<dyn Stream<Item = RagResult<StreamChunk>> + Send + Unpin>> {
        let openai_service = OpenAILLMService::new(LLMConfig {
            provider: LLMProvider::OpenAI,
            ..self.config.clone()
        })?;

        let messages = openai_service.build_rag_prompt(query, &context, conversation_history.as_deref());

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: Some(context),
            stream: true,
            metadata: None,
        };

        self.chat_stream(request).await
    }

    async fn summarize(&self, text: &str, max_length: Option<u32>) -> RagResult<String> {
        let max_length = max_length.unwrap_or(150);

        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("请为以下文本生成一个不超过{}字的摘要。", max_length),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: text.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;
        Ok(response.content)
    }

    async fn extract_keywords(&self, text: &str, count: u32) -> RagResult<Vec<String>> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("请从以下文本中提取{}个最重要的关键词，用逗号分隔。", count),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: text.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;

        let keywords: Vec<String> = response.content
            .split(',')
            .map(|kw| kw.trim().to_string())
            .filter(|kw| !kw.is_empty())
            .collect();

        Ok(keywords)
    }

    async fn suggest_queries(&self, context: &str, count: u32) -> RagResult<Vec<String>> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: format!("基于以下上下文，生成{}个相关的搜索查询建议，每行一个。", count),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: context.to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        let response = self.chat(request).await?;

        let suggestions: Vec<String> = response.content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .take(count as usize)
            .collect();

        Ok(suggestions)
    }

    async fn health_check(&self) -> RagResult<()> {
        let messages = vec![
            ChatMessage {
                role: MessageRole::User,
                content: "Hello".to_string(),
                metadata: None,
            },
        ];

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages,
            context: None,
            stream: false,
            metadata: None,
        };

        self.chat(request).await?;
        Ok(())
    }
}

/// LLM 服务工厂
pub struct LLMServiceFactory;

impl LLMServiceFactory {
    pub fn create_service(config: LLMConfig) -> RagResult<Box<dyn LLMService>> {
        match config.provider {
            LLMProvider::OpenAI => {
                let service = OpenAILLMService::new(config)?;
                Ok(Box::new(service))
            },
            LLMProvider::Anthropic => {
                let service = AnthropicLLMService::new(config)?;
                Ok(Box::new(service))
            },
            _ => Err(RagError::ConfigurationError(format!(
                "不支持的 LLM 提供商: {:?}", config.provider
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config(provider: LLMProvider) -> LLMConfig {
        LLMConfig {
            provider,
            model_name: "test-model".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some("http://localhost:8080".to_string()),
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            timeout_seconds: 30,
            system_prompt: Some("You are a helpful assistant.".to_string()),
        }
    }

    fn create_test_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: MessageRole::System,
                content: "You are a helpful assistant.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "Hello, how are you?".to_string(),
                metadata: None,
            },
        ]
    }

    #[test]
    fn test_llm_config() {
        let config = create_test_config(LLMProvider::OpenAI);
        assert_eq!(config.provider, LLMProvider::OpenAI);
        assert_eq!(config.model_name, "test-model");
        assert!(config.api_key.is_some());
    }

    #[test]
    fn test_chat_message() {
        let message = ChatMessage {
            role: MessageRole::User,
            content: "Test message".to_string(),
            metadata: Some(HashMap::from([
                ("timestamp".to_string(), serde_json::Value::String("2023-01-01".to_string())),
            ])),
        };

        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.content, "Test message");
        assert!(message.metadata.is_some());
    }

    #[test]
    fn test_chat_request() {
        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages: create_test_messages(),
            context: None,
            stream: false,
            metadata: Some(HashMap::from([
                ("session_id".to_string(), serde_json::Value::String("test_session".to_string())),
            ])),
        };

        assert_eq!(request.messages.len(), 2);
        assert!(!request.stream);
        assert!(request.metadata.is_some());
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_stream_chunk() {
        let chunk = StreamChunk {
            id: Uuid::new_v4(),
            content: "Test chunk".to_string(),
            is_final: false,
            finish_reason: None,
        };

        assert_eq!(chunk.content, "Test chunk");
        assert!(!chunk.is_final);
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_message_role_serialization() {
        let role = MessageRole::Assistant;
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: MessageRole = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }

    #[test]
    fn test_llm_provider_serialization() {
        let provider = LLMProvider::OpenAI;
        let serialized = serde_json::to_string(&provider).unwrap();
        let deserialized: LLMProvider = serde_json::from_str(&serialized).unwrap();
        assert_eq!(provider, deserialized);
    }

    #[tokio::test]
    async fn test_openai_service_creation() {
        let config = create_test_config(LLMProvider::OpenAI);
        let service = OpenAILLMService::new(config);
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_anthropic_service_creation() {
        let config = create_test_config(LLMProvider::Anthropic);
        let service = AnthropicLLMService::new(config);
        assert!(service.is_ok());
    }

    #[test]
    fn test_llm_service_factory() {
        let openai_config = create_test_config(LLMProvider::OpenAI);
        let openai_service = LLMServiceFactory::create_service(openai_config);
        assert!(openai_service.is_ok());

        let anthropic_config = create_test_config(LLMProvider::Anthropic);
        let anthropic_service = LLMServiceFactory::create_service(anthropic_config);
        assert!(anthropic_service.is_ok());

        let unsupported_config = create_test_config(LLMProvider::Google);
        let unsupported_service = LLMServiceFactory::create_service(unsupported_config);
        assert!(unsupported_service.is_err());
    }

    #[tokio::test]
    async fn test_build_rag_prompt() {
        let config = create_test_config(LLMProvider::OpenAI);
        let service = OpenAILLMService::new(config).unwrap();

        let context = vec![
            SearchResultItem {
                id: Uuid::new_v4(),
                content: "This is relevant context".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
                document_id: None,
                chunk_index: None,
            }
        ];

        let history = vec![
            ChatMessage {
                role: MessageRole::User,
                content: "Previous question".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: MessageRole::Assistant,
                content: "Previous answer".to_string(),
                metadata: None,
            },
        ];

        let messages = service.build_rag_prompt("New question", &context, Some(&history));

        // 应该包含系统消息、历史对话和新的用户消息
        assert!(messages.len() >= 4);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[messages.len() - 1].role, MessageRole::User);
        assert!(messages[messages.len() - 1].content.contains("New question"));
        assert!(messages[messages.len() - 1].content.contains("relevant context"));
    }

    #[test]
    fn test_chat_response() {
        let response = ChatResponse {
            id: Uuid::new_v4(),
            content: "Test response".to_string(),
            role: MessageRole::Assistant,
            finish_reason: Some("stop".to_string()),
            usage: TokenUsage {
                prompt_tokens: 50,
                completion_tokens: 25,
                total_tokens: 75,
            },
            metadata: None,
        };

        assert_eq!(response.content, "Test response");
        assert_eq!(response.role, MessageRole::Assistant);
        assert_eq!(response.finish_reason, Some("stop".to_string()));
        assert_eq!(response.usage.total_tokens, 75);
    }

    // 注意：以下测试需要真实的 API 密钥才能运行，通常在 CI 环境中跳过
    #[tokio::test]
    #[ignore = "需要真实的 API 密钥"]
    async fn test_openai_integration() {
        let config = LLMConfig {
            provider: LLMProvider::OpenAI,
            model_name: "gpt-3.5-turbo".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            base_url: None,
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            timeout_seconds: 30,
            system_prompt: None,
        };

        if config.api_key.is_none() {
            return; // 跳过测试如果没有 API 密钥
        }

        let service = OpenAILLMService::new(config).unwrap();

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: "Say hello".to_string(),
                    metadata: None,
                }
            ],
            context: None,
            stream: false,
            metadata: None,
        };

        let response = service.chat(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.content.is_empty());
        assert_eq!(response.role, MessageRole::Assistant);
    }

    #[tokio::test]
    #[ignore = "需要真实的 API 密钥"]
    async fn test_anthropic_integration() {
        let config = LLMConfig {
            provider: LLMProvider::Anthropic,
            model_name: "claude-3-sonnet-20240229".to_string(),
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            base_url: None,
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            timeout_seconds: 30,
            system_prompt: None,
        };

        if config.api_key.is_none() {
            return;
        }

        let service = AnthropicLLMService::new(config).unwrap();

        let request = ChatRequest {
            id: Uuid::new_v4(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: "Say hello".to_string(),
                    metadata: None,
                }
            ],
            context: None,
            stream: false,
            metadata: None,
        };

        let response = service.chat(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.content.is_empty());
        assert_eq!(response.role, MessageRole::Assistant);
    }
}