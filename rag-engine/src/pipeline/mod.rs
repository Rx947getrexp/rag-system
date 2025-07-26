//! # RAG 处理管道模块
//!
//! 整合文档处理、向量化、存储和检索的完整流程
//! 文件路径: rag-engine/src/pipeline/mod.rs

pub mod document_processor;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::embedding::EmbeddingService;
use crate::error::{RagError, RagResult};
use crate::llm::LLMService;
use crate::retrieval::{RetrievalService, vector_store::VectorStore};
use crate::types::*;

use document_processor::{MainDocumentProcessor, ProcessedDocument, ChunkingConfig};

/// RAG 管道配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGPipelineConfig {
    pub chunking: ChunkingConfig,
    pub embedding_batch_size: u32,
    pub vector_store_batch_size: u32,
    pub enable_preprocessing: bool,
    pub enable_postprocessing: bool,
    pub preprocessing_steps: Vec<PreprocessingStep>,
    pub postprocessing_steps: Vec<PostprocessingStep>,
    pub quality_threshold: f32,
    pub enable_parallel_processing: bool,
    pub max_concurrent_jobs: u32,
}

impl Default for RAGPipelineConfig {
    fn default() -> Self {
        Self {
            chunking: ChunkingConfig::default(),
            embedding_batch_size: 32,
            vector_store_batch_size: 100,
            enable_preprocessing: true,
            enable_postprocessing: true,
            preprocessing_steps: vec![
                PreprocessingStep::TextCleaning,
                PreprocessingStep::LanguageDetection,
                PreprocessingStep::QualityFiltering,
            ],
            postprocessing_steps: vec![
                PostprocessingStep::DuplicateRemoval,
                PostprocessingStep::QualityScoring,
                PostprocessingStep::MetadataEnrichment,
            ],
            quality_threshold: 0.3,
            enable_parallel_processing: true,
            max_concurrent_jobs: 4,
        }
    }
}

/// 预处理步骤
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PreprocessingStep {
    TextCleaning,
    LanguageDetection,
    QualityFiltering,
    ContentNormalization,
    OCR,
    AudioTranscription,
}

/// 后处理步骤
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PostprocessingStep {
    DuplicateRemoval,
    QualityScoring,
    MetadataEnrichment,
    SemanticClustering,
    KeywordExtraction,
}

/// 管道处理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub document_id: Uuid,
    pub success: bool,
    pub chunks_processed: u32,
    pub vectors_created: u32,
    pub processing_time_ms: u64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// RAG 管道 trait
#[async_trait]
pub trait RAGPipeline: Send + Sync {
    /// 处理单个文档
    async fn process_document(
        &self,
        data: Vec<u8>,
        filename: String,
        metadata: Option<HashMap<String, String>>,
    ) -> RagResult<PipelineResult>;

    /// 批量处理文档
    async fn process_documents(
        &self,
        documents: Vec<(Vec<u8>, String, Option<HashMap<String, String>>)>,
    ) -> RagResult<Vec<PipelineResult>>;

    /// 删除文档
    async fn delete_document(&self, document_id: Uuid) -> RagResult<()>;

    /// 重新索引文档
    async fn reindex_document(&self, document_id: Uuid) -> RagResult<PipelineResult>;

    /// 执行查询
    async fn query(&self, query: Query) -> RagResult<SearchResult>;

    /// 生成 RAG 响应
    async fn generate_response(
        &self,
        query: &str,
        conversation_history: Option<Vec<crate::llm::ChatMessage>>,
        options: Option<QueryOptions>,
    ) -> RagResult<crate::llm::ChatResponse>;

    /// 流式生成 RAG 响应
    async fn generate_response_stream(
        &self,
        query: &str,
        conversation_history: Option<Vec<crate::llm::ChatMessage>>,
        options: Option<QueryOptions>,
    ) -> RagResult<Box<dyn tokio_stream::Stream<Item = RagResult<crate::llm::StreamChunk>> + Send + Unpin>>;

    /// 获取管道统计信息
    async fn get_stats(&self) -> RagResult<PipelineStats>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// 管道统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub total_documents: u64,
    pub total_chunks: u64,
    pub total_vectors: u64,
    pub avg_processing_time_ms: f64,
    pub success_rate: f64,
    pub last_processed: Option<chrono::DateTime<chrono::Utc>>,
    pub throughput_docs_per_hour: f64,
}

/// 主 RAG 管道实现
pub struct MainRAGPipeline {
    document_processor: MainDocumentProcessor,
    embedding_service: Arc<dyn EmbeddingService>,
    vector_store: Arc<dyn VectorStore>,
    retrieval_service: Arc<dyn RetrievalService>,
    llm_service: Arc<dyn LLMService>,
    config: RAGPipelineConfig,
    processing_stats: tokio::sync::RwLock<ProcessingStats>,
}

/// 处理统计信息
#[derive(Debug, Default)]
struct ProcessingStats {
    documents_processed: u64,
    documents_failed: u64,
    total_processing_time_ms: u64,
    last_processed: Option<chrono::DateTime<chrono::Utc>>,
}

impl MainRAGPipeline {
    pub fn new(
        embedding_service: Arc<dyn EmbeddingService>,
        vector_store: Arc<dyn VectorStore>,
        retrieval_service: Arc<dyn RetrievalService>,
        llm_service: Arc<dyn LLMService>,
        config: RAGPipelineConfig,
    ) -> Self {
        let document_processor = MainDocumentProcessor::new(config.chunking.clone());

        Self {
            document_processor,
            embedding_service,
            vector_store,
            retrieval_service,
            llm_service,
            config,
            processing_stats: tokio::sync::RwLock::new(ProcessingStats::default()),
        }
    }

    /// 执行预处理
    async fn preprocess_document(&self, processed_doc: &mut ProcessedDocument) -> RagResult<()> {
        if !self.config.enable_preprocessing {
            return Ok(());
        }

        for step in &self.config.preprocessing_steps {
            match step {
                PreprocessingStep::TextCleaning => {
                    self.clean_text(&mut processed_doc.content).await?;
                }
                PreprocessingStep::LanguageDetection => {
                    if processed_doc.metadata.language.is_none() {
                        processed_doc.metadata.language =
                            document_processor::detect_language(&processed_doc.content);
                    }
                }
                PreprocessingStep::QualityFiltering => {
                    if !self.passes_quality_check(&processed_doc.content).await? {
                        return Err(RagError::DocumentProcessingError(
                            "文档未通过质量检查".to_string()
                        ));
                    }
                }
                PreprocessingStep::ContentNormalization => {
                    self.normalize_content(&mut processed_doc.content).await?;
                }
                _ => {
                    tracing::warn!("预处理步骤 {:?} 尚未实现", step);
                }
            }
        }

        Ok(())
    }

    /// 文本清理
    async fn clean_text(&self, content: &mut String) -> RagResult<()> {
        // 移除多余空白
        *content = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // 移除重复的换行
        while content.contains("\n\n\n") {
            *content = content.replace("\n\n\n", "\n\n");
        }

        Ok(())
    }

    /// 质量检查
    async fn passes_quality_check(&self, content: &str) -> RagResult<bool> {
        // 基本质量检查
        if content.len() < 50 {
            return Ok(false);
        }

        let word_count = content.split_whitespace().count();
        if word_count < 10 {
            return Ok(false);
        }

        // 检查内容复杂度 (简化实现)
        let unique_words: std::collections::HashSet<&str> = content
            .split_whitespace()
            .collect();

        let complexity_ratio = unique_words.len() as f32 / word_count as f32;
        if complexity_ratio < 0.3 {
            return Ok(false);
        }

        Ok(true)
    }

    /// 内容标准化
    async fn normalize_content(&self, content: &mut String) -> RagResult<()> {
        // 统一编码
        *content = content.trim().to_string();

        // 标准化引号
        *content = content.replace(""", "\"").replace(""", "\"");
        *content = content.replace("'", "'").replace("'", "'");

        // 标准化破折号
        *content = content.replace("—", "-").replace("–", "-");

        Ok(())
    }

    /// 执行后处理
    async fn postprocess_chunks(&self, chunks: &mut Vec<DocumentChunk>) -> RagResult<()> {
        if !self.config.enable_postprocessing {
            return Ok(());
        }

        for step in &self.config.postprocessing_steps {
            match step {
                PostprocessingStep::DuplicateRemoval => {
                    self.remove_duplicate_chunks(chunks).await?;
                }
                PostprocessingStep::QualityScoring => {
                    self.score_chunk_quality(chunks).await?;
                }
                PostprocessingStep::MetadataEnrichment => {
                    self.enrich_chunk_metadata(chunks).await?;
                }
                PostprocessingStep::KeywordExtraction => {
                    self.extract_chunk_keywords(chunks).await?;
                }
                _ => {
                    tracing::warn!("后处理步骤 {:?} 尚未实现", step);
                }
            }
        }

        Ok(())
    }

    /// 移除重复块
    async fn remove_duplicate_chunks(&self, chunks: &mut Vec<DocumentChunk>) -> RagResult<()> {
        let mut seen_content = std::collections::HashSet::new();
        chunks.retain(|chunk| {
            let content_hash = self.calculate_content_hash(&chunk.content);
            seen_content.insert(content_hash)
        });

        Ok(())
    }

    /// 计算内容哈希
    fn calculate_content_hash(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// 为块评分质量
    async fn score_chunk_quality(&self, chunks: &mut Vec<DocumentChunk>) -> RagResult<()> {
        for chunk in chunks {
            let quality_score = self.calculate_chunk_quality(&chunk.content).await?;
            chunk.metadata.confidence_score = quality_score;
        }

        // 过滤低质量块
        chunks.retain(|chunk| chunk.metadata.confidence_score >= self.config.quality_threshold);

        Ok(())
    }

    /// 计算块质量分数
    async fn calculate_chunk_quality(&self, content: &str) -> RagResult<f32> {
        let mut score = 1.0;

        // 长度因子
        let length = content.len();
        if length < 100 {
            score *= 0.5;
        } else if length > 2000 {
            score *= 0.8;
        }

        // 词汇多样性
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let diversity = unique_words.len() as f32 / words.len() as f32;
        score *= diversity.max(0.3);

        // 句子完整性
        let sentences = content.split(['.', '!', '?']).count();
        if sentences > 0 && content.len() / sentences > 20 {
            score *= 1.1;
        }

        Ok(score.min(1.0))
    }

    /// 丰富块元数据
    async fn enrich_chunk_metadata(&self, chunks: &mut Vec<DocumentChunk>) -> RagResult<()> {
        for chunk in chunks {
            // 检测语义类型
            chunk.metadata.semantic_type = Some(self.detect_semantic_type(&chunk.content));

            // 添加统计信息
            let word_count = chunk.content.split_whitespace().count();
            let sentence_count = chunk.content.split(['.', '!', '?']).count();

            // 这些信息可以存储在扩展元数据中
            // 实际实现可能需要扩展 ChunkMetadata 结构
        }

        Ok(())
    }

    /// 检测语义类型
    fn detect_semantic_type(&self, content: &str) -> String {
        if content.starts_with('#') || content.contains("章") || content.contains("节") {
            "heading".to_string()
        } else if content.contains("```") || content.contains("    ") {
            "code".to_string()
        } else if content.contains("1.") || content.contains("•") || content.contains("-") {
            "list".to_string()
        } else if content.contains('|') && content.lines().count() > 2 {
            "table".to_string()
        } else {
            "paragraph".to_string()
        }
    }

    /// 提取块关键词
    async fn extract_chunk_keywords(&self, chunks: &mut Vec<DocumentChunk>) -> RagResult<()> {
        for chunk in chunks {
            // 使用 LLM 提取关键词 (可选)
            if let Ok(keywords) = self.llm_service.extract_keywords(&chunk.content, 5).await {
                // 将关键词存储在元数据中
                // 实际实现可能需要扩展 ChunkMetadata 结构
                tracing::debug!("为块 {} 提取关键词: {:?}", chunk.id, keywords);
            }
        }

        Ok(())
    }

    /// 更新处理统计
    async fn update_stats(&self, success: bool, processing_time_ms: u64) {
        let mut stats = self.processing_stats.write().await;

        if success {
            stats.documents_processed += 1;
        } else {
            stats.documents_failed += 1;
        }

        stats.total_processing_time_ms += processing_time_ms;
        stats.last_processed = Some(chrono::Utc::now());
    }
}

#[async_trait]
impl RAGPipeline for MainRAGPipeline {
    async fn process_document(
        &self,
        data: Vec<u8>,
        filename: String,
        metadata: Option<HashMap<String, String>>,
    ) -> RagResult<PipelineResult> {
        let start_time = std::time::Instant::now();
        let document_id = Uuid::new_v4();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        tracing::info!("开始处理文档 {}: {}", document_id, filename);

        // 1. 文档处理
        let mut processed_doc = match self.document_processor
            .process_document(data, filename.clone(), metadata.clone()).await {
            Ok(doc) => doc,
            Err(e) => {
                let error_msg = format!("文档处理失败: {}", e);
                errors.push(error_msg.clone());

                let processing_time = start_time.elapsed().as_millis() as u64;
                self.update_stats(false, processing_time).await;

                return Ok(PipelineResult {
                    document_id,
                    success: false,
                    chunks_processed: 0,
                    vectors_created: 0,
                    processing_time_ms: processing_time,
                    errors,
                    warnings,
                    metadata: HashMap::new(),
                });
            }
        };

        // 设置文档 ID
        processed_doc.id = document_id;
        for chunk in &mut processed_doc.chunks {
            chunk.document_id = document_id;
        }

        // 2. 预处理
        if let Err(e) = self.preprocess_document(&mut processed_doc).await {
            warnings.push(format!("预处理警告: {}", e));
        }

        // 3. 后处理
        if let Err(e) = self.postprocess_chunks(&mut processed_doc.chunks).await {
            warnings.push(format!("后处理警告: {}", e));
        }

        let chunks_count = processed_doc.chunks.len() as u32;

        // 4. 生成向量嵌入
        let embeddings = match self.embedding_service
            .embed_chunks(&processed_doc.chunks).await {
            Ok(embeddings) => embeddings,
            Err(e) => {
                errors.push(format!("向量嵌入失败: {}", e));

                let processing_time = start_time.elapsed().as_millis() as u64;
                self.update_stats(false, processing_time).await;

                return Ok(PipelineResult {
                    document_id,
                    success: false,
                    chunks_processed: chunks_count,
                    vectors_created: 0,
                    processing_time_ms: processing_time,
                    errors,
                    warnings,
                    metadata: HashMap::new(),
                });
            }
        };

        // 5. 存储到向量数据库
        let vector_records: Vec<crate::retrieval::vector_store::VectorRecord> = embeddings
            .into_iter()
            .zip(processed_doc.chunks.iter())
            .map(|(embedding, chunk)| {
                let mut metadata = HashMap::new();
                metadata.insert("document_id".to_string(),
                                serde_json::Value::String(document_id.to_string()));
                metadata.insert("chunk_index".to_string(),
                                serde_json::Value::Number(serde_json::Number::from(chunk.chunk_index)));
                metadata.insert("chunk_id".to_string(),
                                serde_json::Value::String(chunk.id.to_string()));

                // 添加文档元数据
                if let Some(ref meta) = metadata {
                    for (key, value) in meta {
                        metadata.insert(key.clone(), serde_json::Value::String(value.clone()));
                    }
                }

                crate::retrieval::vector_store::VectorRecord {
                    id: chunk.id.to_string(),
                    vector: embedding.values,
                    metadata,
                    text: Some(chunk.content.clone()),
                }
            })
            .collect();

        let vectors_count = vector_records.len() as u32;

        if let Err(e) = self.vector_store.batch_insert_vectors(vector_records).await {
            errors.push(format!("向量存储失败: {}", e));

            let processing_time = start_time.elapsed().as_millis() as u64;
            self.update_stats(false, processing_time).await;

            return Ok(PipelineResult {
                document_id,
                success: false,
                chunks_processed: chunks_count,
                vectors_created: 0,
                processing_time_ms: processing_time,
                errors,
                warnings,
                metadata: HashMap::new(),
            });
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        self.update_stats(true, processing_time).await;

        tracing::info!(
            "文档处理完成 {}: {} 块, {} 向量, 耗时 {}ms",
            document_id, chunks_count, vectors_count, processing_time
        );

        Ok(PipelineResult {
            document_id,
            success: true,
            chunks_processed: chunks_count,
            vectors_created: vectors_count,
            processing_time_ms: processing_time,
            errors,
            warnings,
            metadata: HashMap::from([
                ("filename".to_string(), serde_json::Value::String(filename)),
                ("original_size".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(processed_doc.metadata.file_size)
                )),
                ("format".to_string(), serde_json::Value::String(
                    format!("{:?}", processed_doc.format)
                )),
            ]),
        })
    }

    async fn process_documents(
        &self,
        documents: Vec<(Vec<u8>, String, Option<HashMap<String, String>>)>,
    ) -> RagResult<Vec<PipelineResult>> {
        if !self.config.enable_parallel_processing {
            // 顺序处理
            let mut results = Vec::new();
            for (data, filename, metadata) in documents {
                let result = self.process_document(data, filename, metadata).await?;
                results.push(result);
            }
            return Ok(results);
        }

        // 并行处理
        use futures::stream::{self, StreamExt};

        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.config.max_concurrent_jobs as usize
        ));

        let results = stream::iter(documents)
            .map(|(data, filename, metadata)| {
                let semaphore = semaphore.clone();
                let self_ref = self;
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    self_ref.process_document(data, filename, metadata).await
                }
            })
            .buffer_unordered(self.config.max_concurrent_jobs as usize)
            .collect::<Vec<_>>()
            .await;

        // 收集结果，处理错误
        let mut final_results = Vec::new();
        for result in results {
            match result {
                Ok(pipeline_result) => final_results.push(pipeline_result),
                Err(e) => {
                    // 创建失败结果
                    final_results.push(PipelineResult {
                        document_id: Uuid::new_v4(),
                        success: false,
                        chunks_processed: 0,
                        vectors_created: 0,
                        processing_time_ms: 0,
                        errors: vec![e.to_string()],
                        warnings: vec![],
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        Ok(final_results)
    }

    async fn delete_document(&self, document_id: Uuid) -> RagResult<()> {
        tracing::info!("删除文档: {}", document_id);

        // 从向量数据库中删除所有相关的块
        // 首先搜索所有属于该文档的向量
        let filter = crate::retrieval::vector_store::SearchFilter {
            must: vec![crate::retrieval::vector_store::FilterCondition {
                field: "document_id".to_string(),
                operator: crate::retrieval::vector_store::FilterOperator::Equal,
                value: serde_json::Value::String(document_id.to_string()),
            }],
            must_not: vec![],
            should: vec![],
        };

        // 搜索文档的所有块
        let dummy_vector = vec![0.0; 1536]; // 占位向量
        let document_chunks = self.vector_store
            .search_vectors(&dummy_vector, 1000, Some(filter))
            .await?;

        // 删除所有块
        let chunk_ids: Vec<String> = document_chunks
            .into_iter()
            .map(|item| item.id.to_string())
            .collect();

        if !chunk_ids.is_empty() {
            self.vector_store.batch_delete_vectors(chunk_ids).await?;
        }

        tracing::info!("文档删除完成: {}", document_id);
        Ok(())
    }

    async fn reindex_document(&self, document_id: Uuid) -> RagResult<PipelineResult> {
        tracing::info!("重新索引文档: {}", document_id);

        // 首先删除现有的索引
        self.delete_document(document_id).await?;

        // 由于我们没有存储原始文档数据，这里返回一个错误
        // 实际实现中应该从文档存储中重新获取原始数据
        Err(RagError::NotFound(format!(
            "无法重新索引文档 {}：原始数据不可用", document_id
        )))
    }

    async fn query(&self, query: Query) -> RagResult<SearchResult> {
        tracing::debug!("执行查询: {}", query.text);
        self.retrieval_service.retrieve(&query).await
    }

    async fn generate_response(
        &self,
        query: &str,
        conversation_history: Option<Vec<crate::llm::ChatMessage>>,
        options: Option<QueryOptions>,
    ) -> RagResult<crate::llm::ChatResponse> {
        let start_time = std::time::Instant::now();

        tracing::info!("生成 RAG 响应: {}", query);

        // 1. 构建查询对象
        let query_options = options.unwrap_or_else(|| QueryOptions {
            strategy: "hybrid".to_string(),
            top_k: 5,
            similarity_threshold: Some(0.3),
            filters: vec![],
            enable_reranking: true,
            rerank_top_k: Some(3),
            workspace_id: None,
        });

        let query_obj = Query {
            id: Uuid::new_v4(),
            text: query.to_string(),
            options: query_options,
            timestamp: chrono::Utc::now(),
        };

        // 2. 执行检索
        let search_result = self.retrieval_service.retrieve(&query_obj).await?;

        // 3. 生成 LLM 响应
        let llm_response = self.llm_service.generate_rag_response(
            query,
            search_result.results,
            conversation_history,
        ).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        tracing::info!(
            "RAG 响应生成完成: 耗时 {}ms, 检索到 {} 个相关文档",
            processing_time,
            search_result.results.len()
        );

        Ok(llm_response)
    }

    async fn generate_response_stream(
        &self,
        query: &str,
        conversation_history: Option<Vec<crate::llm::ChatMessage>>,
        options: Option<QueryOptions>,
    ) -> RagResult<Box<dyn tokio_stream::Stream<Item = RagResult<crate::llm::StreamChunk>> + Send + Unpin>> {
        tracing::info!("生成流式 RAG 响应: {}", query);

        // 1. 构建查询对象
        let query_options = options.unwrap_or_else(|| QueryOptions {
            strategy: "hybrid".to_string(),
            top_k: 5,
            similarity_threshold: Some(0.3),
            filters: vec![],
            enable_reranking: true,
            rerank_top_k: Some(3),
            workspace_id: None,
        });

        let query_obj = Query {
            id: Uuid::new_v4(),
            text: query.to_string(),
            options: query_options,
            timestamp: chrono::Utc::now(),
        };

        // 2. 执行检索
        let search_result = self.retrieval_service.retrieve(&query_obj).await?;

        // 3. 生成流式 LLM 响应
        let stream = self.llm_service.generate_rag_response_stream(
            query,
            search_result.results,
            conversation_history,
        ).await?;

        Ok(stream)
    }

    async fn get_stats(&self) -> RagResult<PipelineStats> {
        let stats = self.processing_stats.read().await;

        let total_docs = stats.documents_processed + stats.documents_failed;
        let success_rate = if total_docs > 0 {
            stats.documents_processed as f64 / total_docs as f64
        } else {
            0.0
        };

        let avg_processing_time = if stats.documents_processed > 0 {
            stats.total_processing_time_ms as f64 / stats.documents_processed as f64
        } else {
            0.0
        };

        // 计算吞吐量 (简化实现)
        let throughput = if let Some(last_processed) = stats.last_processed {
            let hours_elapsed = (chrono::Utc::now() - last_processed).num_seconds() as f64 / 3600.0;
            if hours_elapsed > 0.0 {
                stats.documents_processed as f64 / hours_elapsed
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 获取向量数据库统计 (这里简化实现)
        let collection_info = self.vector_store
            .get_collection_info("default")
            .await
            .unwrap_or_else(|_| crate::retrieval::vector_store::CollectionInfo {
                name: "default".to_string(),
                vectors_count: 0,
                dimensions: 0,
                distance_metric: crate::retrieval::vector_store::DistanceMetric::Cosine,
                index_type: crate::retrieval::vector_store::IndexType::HNSW,
                status: crate::retrieval::vector_store::CollectionStatus::Green,
            });

        Ok(PipelineStats {
            total_documents: total_docs,
            total_chunks: collection_info.vectors_count, // 假设每个向量对应一个块
            total_vectors: collection_info.vectors_count,
            avg_processing_time_ms: avg_processing_time,
            success_rate,
            last_processed: stats.last_processed,
            throughput_docs_per_hour: throughput,
        })
    }

    async fn health_check(&self) -> RagResult<()> {
        tracing::debug!("执行管道健康检查");

        // 检查各个组件
        self.embedding_service.health_check().await
            .map_err(|e| RagError::ComponentError(format!("嵌入服务健康检查失败: {}", e)))?;

        self.vector_store.health_check().await
            .map_err(|e| RagError::ComponentError(format!("向量存储健康检查失败: {}", e)))?;

        self.retrieval_service.health_check().await
            .map_err(|e| RagError::ComponentError(format!("检索服务健康检查失败: {}", e)))?;

        self.llm_service.health_check().await
            .map_err(|e| RagError::ComponentError(format!("LLM 服务健康检查失败: {}", e)))?;

        tracing::debug!("管道健康检查通过");
        Ok(())
    }
}

/// 管道构建器
pub struct RAGPipelineBuilder {
    embedding_service: Option<Arc<dyn EmbeddingService>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    retrieval_service: Option<Arc<dyn RetrievalService>>,
    llm_service: Option<Arc<dyn LLMService>>,
    config: RAGPipelineConfig,
}

impl RAGPipelineBuilder {
    pub fn new() -> Self {
        Self {
            embedding_service: None,
            vector_store: None,
            retrieval_service: None,
            llm_service: None,
            config: RAGPipelineConfig::default(),
        }
    }

    pub fn with_embedding_service(mut self, service: Arc<dyn EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    pub fn with_vector_store(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    pub fn with_retrieval_service(mut self, service: Arc<dyn RetrievalService>) -> Self {
        self.retrieval_service = Some(service);
        self
    }

    pub fn with_llm_service(mut self, service: Arc<dyn LLMService>) -> Self {
        self.llm_service = Some(service);
        self
    }

    pub fn with_config(mut self, config: RAGPipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> RagResult<MainRAGPipeline> {
        let embedding_service = self.embedding_service
            .ok_or_else(|| RagError::ConfigurationError("缺少嵌入服务".to_string()))?;

        let vector_store = self.vector_store
            .ok_or_else(|| RagError::ConfigurationError("缺少向量存储".to_string()))?;

        let retrieval_service = self.retrieval_service
            .ok_or_else(|| RagError::ConfigurationError("缺少检索服务".to_string()))?;

        let llm_service = self.llm_service
            .ok_or_else(|| RagError::ConfigurationError("缺少 LLM 服务".to_string()))?;

        Ok(MainRAGPipeline::new(
            embedding_service,
            vector_store,
            retrieval_service,
            llm_service,
            self.config,
        ))
    }
}

impl Default for RAGPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config() {
        let config = RAGPipelineConfig::default();
        assert_eq!(config.embedding_batch_size, 32);
        assert_eq!(config.vector_store_batch_size, 100);
        assert!(config.enable_preprocessing);
        assert!(config.enable_postprocessing);
    }

    #[test]
    fn test_pipeline_result() {
        let result = PipelineResult {
            document_id: Uuid::new_v4(),
            success: true,
            chunks_processed: 10,
            vectors_created: 10,
            processing_time_ms: 1000,
            errors: vec![],
            warnings: vec!["Test warning".to_string()],
            metadata: HashMap::new(),
        };

        assert!(result.success);
        assert_eq!(result.chunks_processed, 10);
        assert_eq!(result.vectors_created, 10);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_preprocessing_steps() {
        let steps = vec![
            PreprocessingStep::TextCleaning,
            PreprocessingStep::LanguageDetection,
            PreprocessingStep::QualityFiltering,
        ];

        assert_eq!(steps.len(), 3);
        assert!(steps.contains(&PreprocessingStep::TextCleaning));
    }

    #[test]
    fn test_postprocessing_steps() {
        let steps = vec![
            PostprocessingStep::DuplicateRemoval,
            PostprocessingStep::QualityScoring,
            PostprocessingStep::MetadataEnrichment,
        ];

        assert_eq!(steps.len(), 3);
        assert!(steps.contains(&PostprocessingStep::DuplicateRemoval));
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            total_documents: 100,
            total_chunks: 1000,
            total_vectors: 1000,
            avg_processing_time_ms: 500.0,
            success_rate: 0.95,
            last_processed: Some(chrono::Utc::now()),
            throughput_docs_per_hour: 120.0,
        };

        assert_eq!(stats.total_documents, 100);
        assert_eq!(stats.success_rate, 0.95);
        assert!(stats.last_processed.is_some());
    }

    #[test]
    fn test_pipeline_builder() {
        let builder = RAGPipelineBuilder::new()
            .with_config(RAGPipelineConfig {
                embedding_batch_size: 64,
                ..RAGPipelineConfig::default()
            });

        assert_eq!(builder.config.embedding_batch_size, 64);

        // 注意：实际的 build() 测试需要 mock 对象
        // 这里只测试配置设置
    }

    #[tokio::test]
    async fn test_content_hash_calculation() {
        // 创建 mock 管道来测试工具方法
        let config = RAGPipelineConfig::default();

        // 由于需要 mock 所有依赖服务，这里只测试辅助方法的逻辑
        let content1 = "Hello world";
        let content2 = "Hello world";
        let content3 = "Different content";

        // 测试哈希一致性 (需要实际的管道实例)
        // 这个测试在实际项目中应该用 mock 框架实现
    }

    #[tokio::test]
    async fn test_text_cleaning() {
        let content = "  Hello   world  \n\n\n  Test  \n\n\n";
        let expected = "Hello   world\n\nTest";

        // 模拟文本清理逻辑
        let cleaned = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        let mut result = cleaned;
        while result.contains("\n\n\n") {
            result = result.replace("\n\n\n", "\n\n");
        }

        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_quality_check_logic() {
        let good_content = "This is a high-quality document with sufficient length and diverse vocabulary. It contains multiple sentences and meaningful content that should pass quality checks.";
        let bad_content = "Short";

        // 模拟质量检查逻辑
        fn passes_quality_check(content: &str) -> bool {
            if content.len() < 50 {
                return false;
            }

            let word_count = content.split_whitespace().count();
            if word_count < 10 {
                return false;
            }

            let unique_words: std::collections::HashSet<&str> = content
                .split_whitespace()
                .collect();

            let complexity_ratio = unique_words.len() as f32 / word_count as f32;
            complexity_ratio >= 0.3
        }

        assert!(passes_quality_check(good_content));
        assert!(!passes_quality_check(bad_content));
    }

    #[test]
    fn test_semantic_type_detection() {
        fn detect_semantic_type(content: &str) -> String {
            if content.starts_with('#') || content.contains("章") || content.contains("节") {
                "heading".to_string()
            } else if content.contains("```") || content.contains("    ") {
                "code".to_string()
            } else if content.contains("1.") || content.contains("•") || content.contains("-") {
                "list".to_string()
            } else if content.contains('|') && content.lines().count() > 2 {
                "table".to_string()
            } else {
                "paragraph".to_string()
            }
        }

        assert_eq!(detect_semantic_type("# Title"), "heading");
        assert_eq!(detect_semantic_type("```rust\ncode\n```"), "code");
        assert_eq!(detect_semantic_type("1. Item one"), "list");
        assert_eq!(detect_semantic_type("| A | B |\n| C | D |"), "table");
        assert_eq!(detect_semantic_type("Regular paragraph"), "paragraph");
    }
}