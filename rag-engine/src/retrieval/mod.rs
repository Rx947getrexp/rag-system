//! # 检索服务模块
//!
//! 实现混合检索策略，包括向量检索、关键词检索和重排序
//! 文件路径: rag-engine/src/retrieval/mod.rs

pub mod vector_store;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::embedding::EmbeddingService;
use crate::error::{RagError, RagResult};
use crate::types::{Query, QueryOptions, SearchResult, SearchResultItem};

use vector_store::{VectorStore, SearchFilter, FilterCondition, FilterOperator};

/// 检索策略
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    VectorOnly,      // 纯向量检索
    KeywordOnly,     // 纯关键词检索
    Hybrid,          // 混合检索
    Semantic,        // 语义检索
    GraphBased,      // 基于图的检索
}

/// 检索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub strategy: RetrievalStrategy,
    pub vector_weight: f32,
    pub keyword_weight: f32,
    pub semantic_weight: f32,
    pub enable_reranking: bool,
    pub rerank_model: Option<String>,
    pub max_candidates: u32,
    pub similarity_threshold: f32,
    pub diversity_threshold: f32,
    pub enable_query_expansion: bool,
    pub enable_filter_optimization: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: RetrievalStrategy::Hybrid,
            vector_weight: 0.7,
            keyword_weight: 0.3,
            semantic_weight: 0.0,
            enable_reranking: true,
            rerank_model: Some("cross-encoder".to_string()),
            max_candidates: 100,
            similarity_threshold: 0.3,
            diversity_threshold: 0.8,
            enable_query_expansion: false,
            enable_filter_optimization: true,
        }
    }
}

/// 检索服务 trait
#[async_trait]
pub trait RetrievalService: Send + Sync {
    /// 执行检索
    async fn retrieve(&self, query: &Query) -> RagResult<SearchResult>;

    /// 获取相似文档
    async fn find_similar(&self, document_id: Uuid, top_k: u32) -> RagResult<Vec<SearchResultItem>>;

    /// 查询建议
    async fn suggest_queries(&self, partial_query: &str, limit: u32) -> RagResult<Vec<String>>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// 主检索服务实现
pub struct MainRetrievalService {
    vector_store: Box<dyn VectorStore>,
    embedding_service: Box<dyn EmbeddingService>,
    keyword_index: Box<dyn KeywordIndex>,
    reranker: Option<Box<dyn Reranker>>,
    config: RetrievalConfig,
    query_cache: tokio::sync::RwLock<HashMap<String, CachedResult>>,
}

/// 缓存结果
#[derive(Debug, Clone)]
struct CachedResult {
    result: SearchResult,
    created_at: chrono::DateTime<chrono::Utc>,
    ttl_seconds: u64,
}

impl CachedResult {
    fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        (now - self.created_at).num_seconds() as u64 > self.ttl_seconds
    }
}

impl MainRetrievalService {
    pub fn new(
        vector_store: Box<dyn VectorStore>,
        embedding_service: Box<dyn EmbeddingService>,
        keyword_index: Box<dyn KeywordIndex>,
        reranker: Option<Box<dyn Reranker>>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            vector_store,
            embedding_service,
            keyword_index,
            reranker,
            config,
            query_cache: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// 生成缓存键
    fn generate_cache_key(&self, query: &Query) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.text.hash(&mut hasher);
        query.options.strategy.hash(&mut hasher);
        query.options.top_k.hash(&mut hasher);

        format!("query_{}_{}", hasher.finish(), query.options.top_k)
    }

    /// 检查缓存
    async fn check_cache(&self, cache_key: &str) -> Option<SearchResult> {
        let cache = self.query_cache.read().await;
        if let Some(cached) = cache.get(cache_key) {
            if !cached.is_expired() {
                return Some(cached.result.clone());
            }
        }
        None
    }

    /// 存储到缓存
    async fn store_cache(&self, cache_key: String, result: SearchResult) {
        let mut cache = self.query_cache.write().await;
        cache.insert(cache_key, CachedResult {
            result,
            created_at: chrono::Utc::now(),
            ttl_seconds: 300, // 5分钟
        });

        // 清理过期缓存
        cache.retain(|_, v| !v.is_expired());
    }

    /// 执行向量检索
    async fn vector_retrieve(&self, query: &Query) -> RagResult<Vec<SearchResultItem>> {
        tracing::debug!("执行向量检索: {}", query.text);

        // 生成查询向量
        let query_embedding = self.embedding_service.embed_text(&query.text).await?;

        // 构建过滤器
        let filter = self.build_search_filter(&query.options)?;

        // 执行向量搜索
        let results = self.vector_store.search_vectors(
            &query_embedding.values,
            self.config.max_candidates,
            filter,
        ).await?;

        // 应用相似度阈值
        let filtered_results: Vec<SearchResultItem> = results
            .into_iter()
            .filter(|item| item.score >= self.config.similarity_threshold)
            .collect();

        tracing::debug!("向量检索返回 {} 个结果", filtered_results.len());
        Ok(filtered_results)
    }

    /// 执行关键词检索
    async fn keyword_retrieve(&self, query: &Query) -> RagResult<Vec<SearchResultItem>> {
        tracing::debug!("执行关键词检索: {}", query.text);

        let results = self.keyword_index.search(
            &query.text,
            self.config.max_candidates,
            &query.options,
        ).await?;

        tracing::debug!("关键词检索返回 {} 个结果", results.len());
        Ok(results)
    }

    /// 执行混合检索
    async fn hybrid_retrieve(&self, query: &Query) -> RagResult<Vec<SearchResultItem>> {
        tracing::debug!("执行混合检索: {}", query.text);

        // 并行执行向量检索和关键词检索
        let (vector_results, keyword_results) = tokio::try_join!(
            self.vector_retrieve(query),
            self.keyword_retrieve(query)
        )?;

        // 合并和重新评分
        let merged_results = self.merge_search_results(
            vector_results,
            keyword_results,
            self.config.vector_weight,
            self.config.keyword_weight,
        );

        tracing::debug!("混合检索返回 {} 个结果", merged_results.len());
        Ok(merged_results)
    }

    /// 合并搜索结果
    fn merge_search_results(
        &self,
        vector_results: Vec<SearchResultItem>,
        keyword_results: Vec<SearchResultItem>,
        vector_weight: f32,
        keyword_weight: f32,
    ) -> Vec<SearchResultItem> {
        let mut result_map: HashMap<Uuid, SearchResultItem> = HashMap::new();

        // 处理向量检索结果
        for mut item in vector_results {
            item.score *= vector_weight;
            result_map.insert(item.id, item);
        }

        // 处理关键词检索结果
        for item in keyword_results {
            let weighted_score = item.score * keyword_weight;

            if let Some(existing) = result_map.get_mut(&item.id) {
                // 合并分数
                existing.score += weighted_score;

                // 合并元数据
                for (key, value) in item.metadata {
                    existing.metadata.entry(key).or_insert(value);
                }
            } else {
                let mut new_item = item;
                new_item.score = weighted_score;
                result_map.insert(new_item.id, new_item);
            }
        }

        // 转换为向量并排序
        let mut results: Vec<SearchResultItem> = result_map.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// 应用多样性过滤
    fn apply_diversity_filter(&self, mut results: Vec<SearchResultItem>) -> Vec<SearchResultItem> {
        if results.len() <= 1 {
            return results;
        }

        let mut diversified = Vec::new();
        diversified.push(results.remove(0)); // 保留最高分的结果

        for candidate in results {
            let mut should_include = true;

            // 检查与已选择结果的相似度
            for selected in &diversified {
                let similarity = self.calculate_content_similarity(&candidate.content, &selected.content);
                if similarity > self.config.diversity_threshold {
                    should_include = false;
                    break;
                }
            }

            if should_include {
                diversified.push(candidate);
            }
        }

        diversified
    }

    /// 计算内容相似度 (简化实现)
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = content1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = content2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// 构建搜索过滤器
    fn build_search_filter(&self, options: &QueryOptions) -> RagResult<Option<SearchFilter>> {
        if options.filters.is_empty() {
            return Ok(None);
        }

        let mut conditions = Vec::new();

        for filter in &options.filters {
            let condition = FilterCondition {
                field: filter.field.clone(),
                operator: match filter.operator.as_str() {
                    "eq" => FilterOperator::Equal,
                    "ne" => FilterOperator::NotEqual,
                    "gt" => FilterOperator::GreaterThan,
                    "gte" => FilterOperator::GreaterThanOrEqual,
                    "lt" => FilterOperator::LessThan,
                    "lte" => FilterOperator::LessThanOrEqual,
                    "in" => FilterOperator::In,
                    "contains" => FilterOperator::Contains,
                    _ => FilterOperator::Equal,
                },
                value: filter.value.clone(),
            };
            conditions.push(condition);
        }

        Ok(Some(SearchFilter {
            must: conditions,
            must_not: vec![],
            should: vec![],
        }))
    }

    /// 执行重排序
    async fn apply_reranking(
        &self,
        query: &str,
        results: Vec<SearchResultItem>,
        top_k: u32,
    ) -> RagResult<Vec<SearchResultItem>> {
        if let Some(reranker) = &self.reranker {
            tracing::debug!("执行重排序，候选数量: {}", results.len());

            let reranked = reranker.rerank(query, results, top_k).await?;

            tracing::debug!("重排序后返回 {} 个结果", reranked.len());
            Ok(reranked)
        } else {
            // 无重排序器，简单截取 top_k
            Ok(results.into_iter().take(top_k as usize).collect())
        }
    }
}

#[async_trait]
impl RetrievalService for MainRetrievalService {
    async fn retrieve(&self, query: &Query) -> RagResult<SearchResult> {
        let start_time = std::time::Instant::now();

        tracing::info!("开始检索: {} (策略: {})", query.text, query.options.strategy);

        // 检查缓存
        let cache_key = self.generate_cache_key(query);
        if let Some(cached_result) = self.check_cache(&cache_key).await {
            tracing::debug!("返回缓存结果");
            return Ok(cached_result);
        }

        // 根据策略执行检索
        let mut results = match self.config.strategy {
            RetrievalStrategy::VectorOnly => self.vector_retrieve(query).await?,
            RetrievalStrategy::KeywordOnly => self.keyword_retrieve(query).await?,
            RetrievalStrategy::Hybrid => self.hybrid_retrieve(query).await?,
            RetrievalStrategy::Semantic => {
                // 语义检索：向量检索 + 语义扩展
                let mut vector_results = self.vector_retrieve(query).await?;

                // TODO: 添加语义扩展逻辑
                vector_results
            },
            RetrievalStrategy::GraphBased => {
                // 基于图的检索：需要图数据库支持
                // 暂时回退到混合检索
                self.hybrid_retrieve(query).await?
            },
        };

        // 应用多样性过滤
        if self.config.diversity_threshold < 1.0 {
            results = self.apply_diversity_filter(results);
        }

        // 执行重排序
        if self.config.enable_reranking {
            results = self.apply_reranking(&query.text, results, query.options.top_k).await?;
        } else {
            results.truncate(query.options.top_k as usize);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        let search_result = SearchResult {
            query: query.clone(),
            results,
            total_found: results.len() as u32,
            processing_time_ms: processing_time,
            strategy_used: self.config.strategy.clone(),
            metadata: HashMap::from([
                ("vector_weight".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(self.config.vector_weight as f64).unwrap()
                )),
                ("keyword_weight".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(self.config.keyword_weight as f64).unwrap()
                )),
                ("reranking_enabled".to_string(), serde_json::Value::Bool(self.config.enable_reranking)),
            ]),
        };

        // 存储到缓存
        self.store_cache(cache_key, search_result.clone()).await;

        tracing::info!(
            "检索完成: {} 个结果, 耗时 {}ms",
            search_result.results.len(),
            processing_time
        );

        Ok(search_result)
    }

    async fn find_similar(&self, document_id: Uuid, top_k: u32) -> RagResult<Vec<SearchResultItem>> {
        tracing::debug!("查找相似文档: {}", document_id);

        // 获取文档的向量表示
        let document_vector = self.vector_store.get_vector(&document_id.to_string()).await?;

        if let Some(doc_vector) = document_vector {
            // 使用文档向量搜索相似文档
            let results = self.vector_store.search_vectors(
                &doc_vector.vector,
                top_k + 1, // +1 因为结果可能包含自身
                None,
            ).await?;

            // 过滤掉自身
            let filtered_results: Vec<SearchResultItem> = results
                .into_iter()
                .filter(|item| item.id != document_id)
                .take(top_k as usize)
                .collect();

            Ok(filtered_results)
        } else {
            Err(RagError::NotFound(format!("文档不存在: {}", document_id)))
        }
    }

    async fn suggest_queries(&self, partial_query: &str, limit: u32) -> RagResult<Vec<String>> {
        tracing::debug!("生成查询建议: {}", partial_query);

        // 简化实现：使用关键词索引生成建议
        let suggestions = self.keyword_index.suggest(partial_query, limit).await?;

        Ok(suggestions)
    }

    async fn health_check(&self) -> RagResult<()> {
        // 检查各个组件的健康状态
        self.vector_store.health_check().await?;
        self.embedding_service.health_check().await?;
        self.keyword_index.health_check().await?;

        if let Some(reranker) = &self.reranker {
            reranker.health_check().await?;
        }

        Ok(())
    }
}

/// 关键词索引 trait
#[async_trait]
pub trait KeywordIndex: Send + Sync {
    /// 添加文档到索引
    async fn add_document(&self, document_id: &str, content: &str, metadata: HashMap<String, serde_json::Value>) -> RagResult<()>;

    /// 批量添加文档
    async fn add_documents(&self, documents: Vec<(String, String, HashMap<String, serde_json::Value>)>) -> RagResult<()>;

    /// 搜索文档
    async fn search(&self, query: &str, limit: u32, options: &QueryOptions) -> RagResult<Vec<SearchResultItem>>;

    /// 生成查询建议
    async fn suggest(&self, partial_query: &str, limit: u32) -> RagResult<Vec<String>>;

    /// 删除文档
    async fn delete_document(&self, document_id: &str) -> RagResult<()>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// 简单的内存关键词索引实现
pub struct InMemoryKeywordIndex {
    documents: tokio::sync::RwLock<HashMap<String, IndexedDocument>>,
    inverted_index: tokio::sync::RwLock<HashMap<String, Vec<String>>>,
}

#[derive(Debug, Clone)]
struct IndexedDocument {
    id: String,
    content: String,
    metadata: HashMap<String, serde_json::Value>,
    tokens: Vec<String>,
}

impl InMemoryKeywordIndex {
    pub fn new() -> Self {
        Self {
            documents: tokio::sync::RwLock::new(HashMap::new()),
            inverted_index: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// 文本分词
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|token| {
                // 简单的标点符号清理
                token.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|token| !token.is_empty() && token.len() > 2)
            .collect()
    }

    /// 计算 TF-IDF 分数
    async fn calculate_tfidf_score(&self, query_tokens: &[String], document: &IndexedDocument) -> f32 {
        let documents = self.documents.read().await;
        let total_docs = documents.len() as f32;
        let mut score = 0.0;

        for query_token in query_tokens {
            // 计算词频 (TF)
            let tf = document.tokens.iter()
                .filter(|token| *token == query_token)
                .count() as f32 / document.tokens.len() as f32;

            if tf > 0.0 {
                // 计算逆文档频率 (IDF)
                let docs_containing_term = documents.values()
                    .filter(|doc| doc.tokens.contains(query_token))
                    .count() as f32;

                let idf = if docs_containing_term > 0.0 {
                    (total_docs / docs_containing_term).ln()
                } else {
                    0.0
                };

                score += tf * idf;
            }
        }

        score
    }
}

#[async_trait]
impl KeywordIndex for InMemoryKeywordIndex {
    async fn add_document(&self, document_id: &str, content: &str, metadata: HashMap<String, serde_json::Value>) -> RagResult<()> {
        let tokens = self.tokenize(content);

        let document = IndexedDocument {
            id: document_id.to_string(),
            content: content.to_string(),
            metadata,
            tokens: tokens.clone(),
        };

        // 添加到文档集合
        {
            let mut documents = self.documents.write().await;
            documents.insert(document_id.to_string(), document);
        }

        // 更新倒排索引
        {
            let mut inverted_index = self.inverted_index.write().await;
            for token in tokens {
                inverted_index.entry(token).or_insert_with(Vec::new).push(document_id.to_string());
            }
        }

        Ok(())
    }

    async fn add_documents(&self, documents: Vec<(String, String, HashMap<String, serde_json::Value>)>) -> RagResult<()> {
        for (id, content, metadata) in documents {
            self.add_document(&id, &content, metadata).await?;
        }
        Ok(())
    }

    async fn search(&self, query: &str, limit: u32, _options: &QueryOptions) -> RagResult<Vec<SearchResultItem>> {
        let query_tokens = self.tokenize(query);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let documents = self.documents.read().await;
        let mut scored_results = Vec::new();

        for document in documents.values() {
            let score = self.calculate_tfidf_score(&query_tokens, document).await;

            if score > 0.0 {
                scored_results.push(SearchResultItem {
                    id: Uuid::parse_str(&document.id).unwrap_or_else(|_| Uuid::new_v4()),
                    content: document.content.clone(),
                    score,
                    metadata: document.metadata.clone(),
                    document_id: Some(Uuid::parse_str(&document.id).unwrap_or_else(|_| Uuid::new_v4())),
                    chunk_index: None,
                });
            }
        }

        // 按分数排序
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored_results.truncate(limit as usize);

        Ok(scored_results)
    }

    async fn suggest(&self, partial_query: &str, limit: u32) -> RagResult<Vec<String>> {
        let partial_tokens = self.tokenize(partial_query);
        if partial_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let inverted_index = self.inverted_index.read().await;
        let mut suggestions = std::collections::HashSet::new();

        for token in &partial_tokens {
            for indexed_token in inverted_index.keys() {
                if indexed_token.starts_with(token) {
                    suggestions.insert(indexed_token.clone());
                }
            }
        }

        let mut result: Vec<String> = suggestions.into_iter().collect();
        result.sort();
        result.truncate(limit as usize);

        Ok(result)
    }

    async fn delete_document(&self, document_id: &str) -> RagResult<()> {
        // 从文档集合中删除
        let document = {
            let mut documents = self.documents.write().await;
            documents.remove(document_id)
        };

        // 从倒排索引中删除
        if let Some(doc) = document {
            let mut inverted_index = self.inverted_index.write().await;
            for token in doc.tokens {
                if let Some(doc_list) = inverted_index.get_mut(&token) {
                    doc_list.retain(|id| id != document_id);
                    if doc_list.is_empty() {
                        inverted_index.remove(&token);
                    }
                }
            }
        }

        Ok(())
    }

    async fn health_check(&self) -> RagResult<()> {
        let documents = self.documents.read().await;
        let index = self.inverted_index.read().await;

        tracing::debug!(
            "关键词索引健康检查: {} 文档, {} 词条",
            documents.len(),
            index.len()
        );

        Ok(())
    }
}

/// 重排序服务 trait
#[async_trait]
pub trait Reranker: Send + Sync {
    /// 重新排序搜索结果
    async fn rerank(&self, query: &str, results: Vec<SearchResultItem>, top_k: u32) -> RagResult<Vec<SearchResultItem>>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;
}

/// 基于交叉编码器的重排序实现
pub struct CrossEncoderReranker {
    model_name: String,
    // 这里应该包含实际的模型推理逻辑
}

impl CrossEncoderReranker {
    pub fn new(model_name: String) -> Self {
        Self { model_name }
    }

    /// 计算查询-文档相关性分数
    async fn calculate_relevance_score(&self, query: &str, document: &str) -> RagResult<f32> {
        // 简化实现：使用基于词汇重叠的相关性分数
        // 实际实现应该使用交叉编码器模型

        let query_words: std::collections::HashSet<&str> = query.split_whitespace().collect();
        let doc_words: std::collections::HashSet<&str> = document.split_whitespace().collect();

        let intersection = query_words.intersection(&doc_words).count();
        let union = query_words.union(&doc_words).count();

        let score = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        Ok(score)
    }
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(&self, query: &str, mut results: Vec<SearchResultItem>, top_k: u32) -> RagResult<Vec<SearchResultItem>> {
        if results.is_empty() {
            return Ok(results);
        }

        // 为每个结果计算重排序分数
        for result in &mut results {
            let relevance_score = self.calculate_relevance_score(query, &result.content).await?;

            // 结合原始分数和相关性分数
            result.score = result.score * 0.5 + relevance_score * 0.5;
        }

        // 重新排序
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k as usize);

        Ok(results)
    }

    async fn health_check(&self) -> RagResult<()> {
        // 简单的健康检查
        tracing::debug!("重排序器健康检查: {}", self.model_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QueryFilter;

    fn create_test_query() -> Query {
        Query {
            id: Uuid::new_v4(),
            text: "test query".to_string(),
            options: QueryOptions {
                strategy: "hybrid".to_string(),
                top_k: 10,
                similarity_threshold: Some(0.5),
                filters: vec![],
                enable_reranking: true,
                rerank_top_k: Some(5),
                workspace_id: None,
            },
            timestamp: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_inmemory_keyword_index() {
        let index = InMemoryKeywordIndex::new();

        // 添加文档
        let metadata = HashMap::new();
        assert!(index.add_document("1", "This is a test document", metadata.clone()).await.is_ok());
        assert!(index.add_document("2", "Another test document with different content", metadata).await.is_ok());

        // 搜索
        let query = create_test_query();
        let results = index.search("test", 10, &query.options).await.unwrap();
        assert_eq!(results.len(), 2);

        // 验证分数排序
        assert!(results[0].score >= results[1].score);

        // 生成建议
        let suggestions = index.suggest("te", 5).await.unwrap();
        assert!(!suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_cross_encoder_reranker() {
        let reranker = CrossEncoderReranker::new("test-model".to_string());

        let results = vec![
            SearchResultItem {
                id: Uuid::new_v4(),
                content: "This document is about artificial intelligence".to_string(),
                score: 0.8,
                metadata: HashMap::new(),
                document_id: None,
                chunk_index: None,
            },
            SearchResultItem {
                id: Uuid::new_v4(),
                content: "This is a cooking recipe".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
                document_id: None,
                chunk_index: None,
            },
        ];

        let reranked = reranker.rerank("artificial intelligence", results, 2).await.unwrap();
        assert_eq!(reranked.len(), 2);

        // 第一个结果应该更相关
        assert!(reranked[0].content.contains("artificial intelligence"));
    }

    #[test]
    fn test_retrieval_config() {
        let config = RetrievalConfig::default();
        assert_eq!(config.strategy, RetrievalStrategy::Hybrid);
        assert_eq!(config.vector_weight, 0.7);
        assert_eq!(config.keyword_weight, 0.3);
        assert!(config.enable_reranking);
    }

    #[test]
    fn test_cached_result() {
        let result = SearchResult {
            query: create_test_query(),
            results: vec![],
            total_found: 0,
            processing_time_ms: 100,
            strategy_used: RetrievalStrategy::Hybrid,
            metadata: HashMap::new(),
        };

        let cached = CachedResult {
            result,
            created_at: chrono::Utc::now(),
            ttl_seconds: 300,
        };

        assert!(!cached.is_expired());

        let expired_cached = CachedResult {
            result: cached.result.clone(),
            created_at: chrono::Utc::now() - chrono::Duration::seconds(400),
            ttl_seconds: 300,
        };

        assert!(expired_cached.is_expired());
    }

    #[test]
    fn test_content_similarity() {
        let service = create_mock_retrieval_service();

        let content1 = "artificial intelligence machine learning";
        let content2 = "machine learning deep learning";
        let content3 = "cooking recipe ingredients";

        let sim1 = service.calculate_content_similarity(content1, content2);
        let sim2 = service.calculate_content_similarity(content1, content3);

        assert!(sim1 > sim2); // content1 和 content2 更相似
    }

    fn create_mock_retrieval_service() -> MainRetrievalService {
        // 这里需要创建 mock 对象，实际测试中应该使用 mock 框架
        // 简化实现用于演示
        panic!("需要实现 mock 对象创建")
    }
}