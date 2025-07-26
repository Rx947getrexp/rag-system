//! # 嵌入向量生成模块
//!
//! 负责将文本转换为向量表示，支持多种嵌入模型
//! 文件路径: rag-engine/src/embedding/mod.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{RagError, RagResult};
use crate::types::{DocumentChunk, EmbeddingVector};

/// 嵌入模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub provider: EmbeddingProvider,
    pub dimensions: u32,
    pub max_tokens: u32,
    pub batch_size: u32,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model_params: HashMap<String, serde_json::Value>,
}

/// 嵌入提供商
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    OpenAI,
    HuggingFace,
    Cohere,
    Anthropic,
    Local,
    Ollama,
}

/// 嵌入请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub id: Uuid,
    pub texts: Vec<String>,
    pub model: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// 嵌入响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub id: Uuid,
    pub embeddings: Vec<EmbeddingVector>,
    pub model: String,
    pub usage: EmbeddingUsage,
    pub processing_time_ms: u64,
}

/// 使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub total_tokens: u32,
    pub prompt_tokens: u32,
    pub total_requests: u32,
}

/// 嵌入服务 trait
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// 生成单个文本的嵌入向量
    async fn embed_text(&self, text: &str) -> RagResult<EmbeddingVector>;

    /// 批量生成嵌入向量
    async fn embed_batch(&self, texts: &[String]) -> RagResult<Vec<EmbeddingVector>>;

    /// 生成文档块的嵌入向量
    async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> RagResult<Vec<EmbeddingVector>>;

    /// 获取模型信息
    fn get_model_info(&self) -> &EmbeddingConfig;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// 主嵌入服务管理器
pub struct EmbeddingManager {
    services: RwLock<HashMap<String, Box<dyn EmbeddingService>>>,
    default_service: String,
}

impl EmbeddingManager {
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            default_service: "default".to_string(),
        }
    }

    /// 注册嵌入服务
    pub async fn register_service(&self, name: String, service: Box<dyn EmbeddingService>) {
        let mut services = self.services.write().await;
        services.insert(name, service);
    }

    /// 设置默认服务
    pub fn set_default_service(&mut self, name: String) {
        self.default_service = name;
    }

    /// 获取服务
    pub async fn get_service(&self, name: Option<&str>) -> RagResult<&dyn EmbeddingService> {
        let service_name = name.unwrap_or(&self.default_service);
        let services = self.services.read().await;

        // 注意：这里有生命周期问题，实际实现中需要使用 Arc<dyn EmbeddingService>
        // 为了演示，我们返回一个错误
        Err(RagError::EmbeddingError("服务获取方法需要重构".to_string()))
    }

    /// 处理嵌入请求
    pub async fn process_request(&self, request: EmbeddingRequest) -> RagResult<EmbeddingResponse> {
        let start_time = std::time::Instant::now();

        // 这里需要重构以支持正确的生命周期管理
        let service = self.get_default_service().await?;
        let embeddings = service.embed_batch(&request.texts).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(EmbeddingResponse {
            id: request.id,
            embeddings,
            model: request.model,
            usage: EmbeddingUsage {
                total_tokens: request.texts.iter().map(|t| t.len() as u32).sum(),
                prompt_tokens: request.texts.len() as u32,
                total_requests: 1,
            },
            processing_time_ms: processing_time,
        })
    }

    async fn get_default_service(&self) -> RagResult<&dyn EmbeddingService> {
        // 临时实现
        Err(RagError::EmbeddingError("需要实现服务获取".to_string()))
    }
}

/// OpenAI 嵌入服务
pub struct OpenAIEmbeddingService {
    config: EmbeddingConfig,
    client: reqwest::Client,
}

impl OpenAIEmbeddingService {
    pub fn new(config: EmbeddingConfig) -> RagResult<Self> {
        if config.provider != EmbeddingProvider::OpenAI {
            return Err(RagError::ConfigurationError(
                "配置提供商必须是 OpenAI".to_string()
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| RagError::EmbeddingError(format!("HTTP 客户端创建失败: {}", e)))?;

        Ok(Self { config, client })
    }

    async fn call_openai_api(&self, texts: &[String]) -> RagResult<Vec<Vec<f32>>> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| RagError::ConfigurationError("缺少 OpenAI API 密钥".to_string()))?;

        let base_url = self.config.base_url.as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let request_body = serde_json::json!({
            "input": texts,
            "model": self.config.model_name,
            "encoding_format": "float"
        });

        let response = self.client
            .post(&format!("{}/embeddings", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RagError::EmbeddingError(format!("API 请求失败: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(RagError::EmbeddingError(format!(
                "OpenAI API 错误: {}", error_text
            )));
        }

        let response_data: serde_json::Value = response.json().await
            .map_err(|e| RagError::EmbeddingError(format!("响应解析失败: {}", e)))?;

        let embeddings = response_data["data"]
            .as_array()
            .ok_or_else(|| RagError::EmbeddingError("无效的响应格式".to_string()))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| RagError::EmbeddingError("无效的嵌入格式".to_string()))?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect::<Result<Vec<Vec<f32>>, RagError>>()?;

        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingService for OpenAIEmbeddingService {
    async fn embed_text(&self, text: &str) -> RagResult<EmbeddingVector> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed_batch(&texts).await?;

        embeddings.pop()
            .ok_or_else(|| RagError::EmbeddingError("空的嵌入响应".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> RagResult<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // 分批处理
        let mut all_embeddings = Vec::new();
        let batch_size = self.config.batch_size as usize;

        for chunk in texts.chunks(batch_size) {
            let raw_embeddings = self.call_openai_api(chunk).await?;

            for (i, embedding) in raw_embeddings.into_iter().enumerate() {
                let vector = EmbeddingVector {
                    id: Uuid::new_v4(),
                    values: embedding,
                    metadata: Some(HashMap::from([
                        ("text".to_string(), serde_json::Value::String(chunk[i].clone())),
                        ("model".to_string(), serde_json::Value::String(self.config.model_name.clone())),
                        ("provider".to_string(), serde_json::Value::String("openai".to_string())),
                    ])),
                };
                all_embeddings.push(vector);
            }

            // 避免 API 限流
            if texts.len() > batch_size {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        Ok(all_embeddings)
    }

    async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> RagResult<Vec<EmbeddingVector>> {
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
        let mut embeddings = self.embed_batch(&texts).await?;

        // 添加块相关的元数据
        for (i, embedding) in embeddings.iter_mut().enumerate() {
            if let Some(chunk) = chunks.get(i) {
                if let Some(metadata) = &mut embedding.metadata {
                    metadata.insert("chunk_id".to_string(),
                                    serde_json::Value::String(chunk.id.to_string()));
                    metadata.insert("document_id".to_string(),
                                    serde_json::Value::String(chunk.document_id.to_string()));
                    metadata.insert("chunk_index".to_string(),
                                    serde_json::Value::Number(serde_json::Number::from(chunk.chunk_index)));
                }
            }
        }

        Ok(embeddings)
    }

    fn get_model_info(&self) -> &EmbeddingConfig {
        &self.config
    }

    async fn health_check(&self) -> RagResult<()> {
        // 使用小的测试文本检查 API 连接
        let test_text = "health check";
        self.embed_text(test_text).await?;
        Ok(())
    }
}

/// HuggingFace 嵌入服务
pub struct HuggingFaceEmbeddingService {
    config: EmbeddingConfig,
    client: reqwest::Client,
}

impl HuggingFaceEmbeddingService {
    pub fn new(config: EmbeddingConfig) -> RagResult<Self> {
        if config.provider != EmbeddingProvider::HuggingFace {
            return Err(RagError::ConfigurationError(
                "配置提供商必须是 HuggingFace".to_string()
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| RagError::EmbeddingError(format!("HTTP 客户端创建失败: {}", e)))?;

        Ok(Self { config, client })
    }

    async fn call_huggingface_api(&self, texts: &[String]) -> RagResult<Vec<Vec<f32>>> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| RagError::ConfigurationError("缺少 HuggingFace API 密钥".to_string()))?;

        let base_url = self.config.base_url.as_deref()
            .unwrap_or("https://api-inference.huggingface.co");

        let url = format!("{}/pipeline/feature-extraction/{}", base_url, self.config.model_name);

        let request_body = serde_json::json!({
            "inputs": texts,
            "options": {
                "wait_for_model": true
            }
        });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RagError::EmbeddingError(format!("API 请求失败: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(RagError::EmbeddingError(format!(
                "HuggingFace API 错误: {}", error_text
            )));
        }

        let embeddings: Vec<Vec<f32>> = response.json().await
            .map_err(|e| RagError::EmbeddingError(format!("响应解析失败: {}", e)))?;

        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingService for HuggingFaceEmbeddingService {
    async fn embed_text(&self, text: &str) -> RagResult<EmbeddingVector> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed_batch(&texts).await?;

        embeddings.pop()
            .ok_or_else(|| RagError::EmbeddingError("空的嵌入响应".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> RagResult<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let raw_embeddings = self.call_huggingface_api(texts).await?;
        let mut embeddings = Vec::new();

        for (i, embedding) in raw_embeddings.into_iter().enumerate() {
            let vector = EmbeddingVector {
                id: Uuid::new_v4(),
                values: embedding,
                metadata: Some(HashMap::from([
                    ("text".to_string(), serde_json::Value::String(texts[i].clone())),
                    ("model".to_string(), serde_json::Value::String(self.config.model_name.clone())),
                    ("provider".to_string(), serde_json::Value::String("huggingface".to_string())),
                ])),
            };
            embeddings.push(vector);
        }

        Ok(embeddings)
    }

    async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> RagResult<Vec<EmbeddingVector>> {
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
        let mut embeddings = self.embed_batch(&texts).await?;

        // 添加块相关的元数据
        for (i, embedding) in embeddings.iter_mut().enumerate() {
            if let Some(chunk) = chunks.get(i) {
                if let Some(metadata) = &mut embedding.metadata {
                    metadata.insert("chunk_id".to_string(),
                                    serde_json::Value::String(chunk.id.to_string()));
                    metadata.insert("document_id".to_string(),
                                    serde_json::Value::String(chunk.document_id.to_string()));
                    metadata.insert("chunk_index".to_string(),
                                    serde_json::Value::Number(serde_json::Number::from(chunk.chunk_index)));
                }
            }
        }

        Ok(embeddings)
    }

    fn get_model_info(&self) -> &EmbeddingConfig {
        &self.config
    }

    async fn health_check(&self) -> RagResult<()> {
        let test_text = "health check";
        self.embed_text(test_text).await?;
        Ok(())
    }
}

/// 本地嵌入服务 (使用 ONNX 或其他本地模型)
pub struct LocalEmbeddingService {
    config: EmbeddingConfig,
    // 这里可以添加 ONNX 运行时或其他本地推理引擎
}

impl LocalEmbeddingService {
    pub fn new(config: EmbeddingConfig) -> RagResult<Self> {
        if config.provider != EmbeddingProvider::Local {
            return Err(RagError::ConfigurationError(
                "配置提供商必须是 Local".to_string()
            ));
        }

        // TODO: 初始化本地模型
        tracing::warn!("本地嵌入服务尚未完全实现");

        Ok(Self { config })
    }

    async fn generate_embeddings_locally(&self, texts: &[String]) -> RagResult<Vec<Vec<f32>>> {
        // TODO: 实现本地嵌入生成
        // 这里应该加载 ONNX 模型或其他本地推理引擎

        // 临时实现：返回随机向量用于测试
        let mut embeddings = Vec::new();
        for _ in texts {
            let mut embedding = Vec::new();
            for _ in 0..self.config.dimensions {
                embedding.push(rand::random::<f32>() * 2.0 - 1.0); // [-1, 1]
            }
            embeddings.push(embedding);
        }

        tracing::warn!("使用随机向量作为本地嵌入 (仅用于测试)");
        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingService for LocalEmbeddingService {
    async fn embed_text(&self, text: &str) -> RagResult<EmbeddingVector> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed_batch(&texts).await?;

        embeddings.pop()
            .ok_or_else(|| RagError::EmbeddingError("空的嵌入响应".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> RagResult<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let raw_embeddings = self.generate_embeddings_locally(texts).await?;
        let mut embeddings = Vec::new();

        for (i, embedding) in raw_embeddings.into_iter().enumerate() {
            let vector = EmbeddingVector {
                id: Uuid::new_v4(),
                values: embedding,
                metadata: Some(HashMap::from([
                    ("text".to_string(), serde_json::Value::String(texts[i].clone())),
                    ("model".to_string(), serde_json::Value::String(self.config.model_name.clone())),
                    ("provider".to_string(), serde_json::Value::String("local".to_string())),
                ])),
            };
            embeddings.push(vector);
        }

        Ok(embeddings)
    }

    async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> RagResult<Vec<EmbeddingVector>> {
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
        self.embed_batch(&texts).await
    }

    fn get_model_info(&self) -> &EmbeddingConfig {
        &self.config
    }

    async fn health_check(&self) -> RagResult<()> {
        // 本地服务健康检查
        Ok(())
    }
}

/// 嵌入向量工具函数
pub mod utils {
    use super::*;

    /// 计算两个向量的余弦相似度
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// 计算向量的 L2 范数
    pub fn l2_norm(vector: &[f32]) -> f32 {
        vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// 归一化向量
    pub fn normalize_vector(vector: &mut [f32]) {
        let norm = l2_norm(vector);
        if norm > 0.0 {
            for value in vector {
                *value /= norm;
            }
        }
    }

    /// 计算向量间的欧氏距离
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// 批量计算相似度
    pub fn batch_cosine_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
        candidates
            .iter()
            .map(|candidate| cosine_similarity(query, candidate))
            .collect()
    }

    /// 找到最相似的 top-k 向量
    pub fn find_top_k_similar(
        query: &[f32],
        candidates: &[Vec<f32>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| (i, cosine_similarity(query, candidate)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// 验证向量维度
    pub fn validate_dimensions(vectors: &[Vec<f32>]) -> RagResult<u32> {
        if vectors.is_empty() {
            return Err(RagError::EmbeddingError("空向量列表".to_string()));
        }

        let expected_dim = vectors[0].len() as u32;
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() as u32 != expected_dim {
                return Err(RagError::EmbeddingError(format!(
                    "向量 {} 维度不匹配: 期望 {}, 得到 {}",
                    i, expected_dim, vector.len()
                )));
            }
        }

        Ok(expected_dim)
    }

    /// 计算向量集合的中心点
    pub fn compute_centroid(vectors: &[Vec<f32>]) -> RagResult<Vec<f32>> {
        if vectors.is_empty() {
            return Err(RagError::EmbeddingError("空向量列表".to_string()));
        }

        let dimensions = validate_dimensions(vectors)?;
        let mut centroid = vec![0.0; dimensions as usize];

        for vector in vectors {
            for (i, &value) in vector.iter().enumerate() {
                centroid[i] += value;
            }
        }

        let count = vectors.len() as f32;
        for value in &mut centroid {
            *value /= count;
        }

        Ok(centroid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;

    fn create_test_config(provider: EmbeddingProvider) -> EmbeddingConfig {
        EmbeddingConfig {
            model_name: "test-model".to_string(),
            provider,
            dimensions: 128,
            max_tokens: 8192,
            batch_size: 10,
            api_key: Some("test-key".to_string()),
            base_url: Some("http://localhost:8080".to_string()),
            model_params: HashMap::new(),
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&a, &c), 1.0);
    }

    #[test]
    fn test_l2_norm() {
        let vector = vec![3.0, 4.0];
        assert_eq!(l2_norm(&vector), 5.0);
    }

    #[test]
    fn test_normalize_vector() {
        let mut vector = vec![3.0, 4.0];
        normalize_vector(&mut vector);
        assert!((l2_norm(&vector) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(euclidean_distance(&a, &b), 5.0);
    }

    #[test]
    fn test_find_top_k_similar() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0],  // 完全匹配
            vec![0.0, 1.0],  // 垂直
            vec![0.7, 0.7],  // 45度角
        ];

        let results = find_top_k_similar(&query, &candidates, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // 最相似的是索引 0
        assert!(results[0].1 > results[1].1); // 第一个分数更高
    }

    #[test]
    fn test_validate_dimensions() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        assert_eq!(validate_dimensions(&vectors).unwrap(), 3);

        let mismatched_vectors = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0],
        ];
        assert!(validate_dimensions(&mismatched_vectors).is_err());
    }

    #[test]
    fn test_compute_centroid() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![2.0, 2.0],
        ];
        let centroid = compute_centroid(&vectors).unwrap();
        assert_eq!(centroid, vec![1.0, 1.0]);
    }

    #[tokio::test]
    async fn test_local_embedding_service() {
        let config = create_test_config(EmbeddingProvider::Local);
        let service = LocalEmbeddingService::new(config).unwrap();

        let text = "Hello, world!";
        let embedding = service.embed_text(text).await.unwrap();

        assert_eq!(embedding.values.len(), 128);
        assert!(embedding.metadata.is_some());
    }

    #[tokio::test]
    async fn test_batch_embedding() {
        let config = create_test_config(EmbeddingProvider::Local);
        let service = LocalEmbeddingService::new(config).unwrap();

        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let embeddings = service.embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);

        for embedding in &embeddings {
            assert_eq!(embedding.values.len(), 128);
            assert!(embedding.metadata.is_some());
        }
    }

    #[tokio::test]
    async fn test_chunk_embedding() {
        let config = create_test_config(EmbeddingProvider::Local);
        let service = LocalEmbeddingService::new(config).unwrap();

        let chunks = vec![
            DocumentChunk {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                content: "First chunk content".to_string(),
                chunk_index: 0,
                start_char: 0,
                end_char: 19,
                metadata: crate::pipeline::document_processor::ChunkMetadata {
                    section_title: None,
                    page_number: None,
                    paragraph_index: None,
                    semantic_type: None,
                    confidence_score: 1.0,
                },
            },
            DocumentChunk {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                content: "Second chunk content".to_string(),
                chunk_index: 1,
                start_char: 20,
                end_char: 40,
                metadata: crate::pipeline::document_processor::ChunkMetadata {
                    section_title: None,
                    page_number: None,
                    paragraph_index: None,
                    semantic_type: None,
                    confidence_score: 1.0,
                },
            },
        ];

        let embeddings = service.embed_chunks(&chunks).await.unwrap();
        assert_eq!(embeddings.len(), 2);

        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(embedding.values.len(), 128);

            if let Some(metadata) = &embedding.metadata {
                assert!(metadata.contains_key("chunk_id"));
                assert!(metadata.contains_key("document_id"));
                assert!(metadata.contains_key("chunk_index"));
            }
        }
    }

    #[test]
    fn test_embedding_config() {
        let config = EmbeddingConfig {
            model_name: "text-embedding-ada-002".to_string(),
            provider: EmbeddingProvider::OpenAI,
            dimensions: 1536,
            max_tokens: 8191,
            batch_size: 100,
            api_key: Some("sk-test123".to_string()),
            base_url: None,
            model_params: HashMap::from([
                ("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.0).unwrap())),
            ]),
        };

        assert_eq!(config.provider, EmbeddingProvider::OpenAI);
        assert_eq!(config.dimensions, 1536);
        assert!(config.api_key.is_some());
    }

    #[test]
    fn test_embedding_request_response() {
        let request = EmbeddingRequest {
            id: Uuid::new_v4(),
            texts: vec!["test text".to_string()],
            model: "test-model".to_string(),
            metadata: Some(HashMap::from([
                ("source".to_string(), serde_json::Value::String("test".to_string())),
            ])),
        };

        assert_eq!(request.texts.len(), 1);
        assert!(request.metadata.is_some());

        let response = EmbeddingResponse {
            id: request.id,
            embeddings: vec![EmbeddingVector {
                id: Uuid::new_v4(),
                values: vec![0.1, 0.2, 0.3],
                metadata: None,
            }],
            model: "test-model".to_string(),
            usage: EmbeddingUsage {
                total_tokens: 10,
                prompt_tokens: 10,
                total_requests: 1,
            },
            processing_time_ms: 100,
        };

        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn test_embedding_manager() {
        let manager = EmbeddingManager::new();

        let config = create_test_config(EmbeddingProvider::Local);
        let service = Box::new(LocalEmbeddingService::new(config).unwrap());

        manager.register_service("test".to_string(), service).await;

        // 注意：由于生命周期问题，实际的服务获取测试需要重构
        // 这里主要测试管理器的基本结构
    }

    #[test]
    fn test_embedding_provider_serialization() {
        let provider = EmbeddingProvider::OpenAI;
        let serialized = serde_json::to_string(&provider).unwrap();
        let deserialized: EmbeddingProvider = serde_json::from_str(&serialized).unwrap();
        assert_eq!(provider, deserialized);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];

        let similarities = batch_cosine_similarity(&query, &candidates);
        assert_eq!(similarities.len(), 3);
        assert_eq!(similarities[0], 1.0);  // 完全匹配
        assert_eq!(similarities[1], 0.0);  // 垂直
        assert_eq!(similarities[2], -1.0); // 相反
    }
}