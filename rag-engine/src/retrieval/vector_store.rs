//! # 向量数据库集成模块
//!
//! 支持多种向量数据库，包括 Qdrant、Pinecone、Weaviate 等
//! 文件路径: rag-engine/src/retrieval/vector_store.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{RagError, RagResult};
use crate::types::{EmbeddingVector, SearchResult, SearchResultItem};

/// 向量存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub provider: VectorStoreProvider,
    pub connection_string: String,
    pub collection_name: String,
    pub dimensions: u32,
    pub distance_metric: DistanceMetric,
    pub index_config: IndexConfig,
    pub batch_size: u32,
    pub timeout_seconds: u32,
}

/// 向量存储提供商
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorStoreProvider {
    Qdrant,
    Pinecone,
    Weaviate,
    Chroma,
    Milvus,
    InMemory,
}

/// 距离度量方式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Dot,
    Manhattan,
}

/// 索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub ef_construct: Option<u32>,
    pub m: Option<u32>,
    pub quantization: Option<QuantizationConfig>,
}

/// 索引类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    HNSW,
    IVF,
    Flat,
    LSH,
}

/// 量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub compression_ratio: f32,
}

/// 量化类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizationType {
    Int8,
    Binary,
    Product,
}

/// 向量存储操作 trait
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// 创建集合/索引
    async fn batch_insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // 分批处理大量向量
        let batch_size = self.config.batch_size as usize;
        for chunk in vectors.chunks(batch_size) {
            self.insert_vectors(chunk.to_vec()).await?;
        }

        Ok(())
    }

    async fn update_vector(&self, id: &str, vector: VectorRecord) -> RagResult<()> {
        self.insert_vectors(vec![vector]).await
    }

    async fn delete_vector(&self, id: &str) -> RagResult<()> {
        use qdrant_client::qdrant::{DeletePoints, PointsSelector, points_selector::PointsSelectorOneOf};

        let delete_points = DeletePoints {
            collection_name: self.config.collection_name.clone(),
            wait: Some(true),
            points: Some(PointsSelector {
                points_selector_one_of: Some(PointsSelectorOneOf::Points(
                    qdrant_client::qdrant::PointsIdsList {
                        ids: vec![qdrant_client::qdrant::PointId {
                            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.to_string())),
                        }],
                    },
                )),
            }),
            ordering: None,
        };

        self.client
            .delete_points(&delete_points)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("删除向量失败: {}", e)))?;

        Ok(())
    }

    async fn batch_delete_vectors(&self, ids: Vec<String>) -> RagResult<()> {
        if ids.is_empty() {
            return Ok(());
        }

        use qdrant_client::qdrant::{DeletePoints, PointsSelector, points_selector::PointsSelectorOneOf};

        let point_ids: Vec<qdrant_client::qdrant::PointId> = ids
            .into_iter()
            .map(|id| qdrant_client::qdrant::PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id)),
            })
            .collect();

        let delete_points = DeletePoints {
            collection_name: self.config.collection_name.clone(),
            wait: Some(true),
            points: Some(PointsSelector {
                points_selector_one_of: Some(PointsSelectorOneOf::Points(
                    qdrant_client::qdrant::PointsIdsList {
                        ids: point_ids,
                    },
                )),
            }),
            ordering: None,
        };

        self.client
            .delete_points(&delete_points)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("批量删除向量失败: {}", e)))?;

        Ok(())
    }

    async fn get_vector(&self, id: &str) -> RagResult<Option<VectorRecord>> {
        use qdrant_client::qdrant::{GetPoints, PointsSelector, points_selector::PointsSelectorOneOf};

        let get_points = GetPoints {
            collection_name: self.config.collection_name.clone(),
            ids: Some(PointsSelector {
                points_selector_one_of: Some(PointsSelectorOneOf::Points(
                    qdrant_client::qdrant::PointsIdsList {
                        ids: vec![qdrant_client::qdrant::PointId {
                            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.to_string())),
                        }],
                    },
                )),
            }),
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            read_consistency: None,
        };

        let response = self.client
            .get_points(&get_points)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("获取向量失败: {}", e)))?;

        if let Some(point) = response.result.into_iter().next() {
            let vector_data = if let Some(vectors) = point.vectors {
                match vectors.vectors_options {
                    Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(v)) => v.data,
                    _ => return Err(RagError::VectorStoreError("无效的向量格式".to_string())),
                }
            } else {
                return Err(RagError::VectorStoreError("向量数据为空".to_string()));
            };

            let mut metadata = HashMap::new();
            let mut text = None;

            for (key, value) in point.payload {
                if key == "text" {
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(s)) = value.kind {
                        text = Some(s);
                    }
                } else {
                    let json_value = match value.kind {
                        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                            serde_json::Value::String(s)
                        },
                        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
                            serde_json::Value::Number(serde_json::Number::from(i))
                        },
                        Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)) => {
                            serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap_or_default())
                        },
                        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => {
                            serde_json::Value::Bool(b)
                        },
                        _ => continue,
                    };
                    metadata.insert(key, json_value);
                }
            }

            Ok(Some(VectorRecord {
                id: id.to_string(),
                vector: vector_data,
                metadata,
                text,
            }))
        } else {
            Ok(None)
        }
    }

    async fn search_vectors(
        &self,
        query_vector: &[f32],
        limit: u32,
        filter: Option<SearchFilter>,
    ) -> RagResult<Vec<SearchResultItem>> {
        use qdrant_client::qdrant::{SearchPoints, Vector};

        let qdrant_filter = filter.map(|f| self.convert_search_filter(&f));

        let search_points = SearchPoints {
            collection_name: self.config.collection_name.clone(),
            vector: query_vector.to_vec(),
            filter: qdrant_filter,
            limit: limit as u64,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            with_vectors: Some(false.into()),
            read_consistency: None,
        };

        let response = self.client
            .search_points(&search_points)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("向量搜索失败: {}", e)))?;

        let mut results = Vec::new();
        for scored_point in response.result {
            let mut metadata = HashMap::new();
            let mut content = String::new();

            for (key, value) in scored_point.payload {
                if key == "text" {
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(s)) = value.kind {
                        content = s;
                    }
                } else {
                    let json_value = match value.kind {
                        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                            serde_json::Value::String(s)
                        },
                        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
                            serde_json::Value::Number(serde_json::Number::from(i))
                        },
                        Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)) => {
                            serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap_or_default())
                        },
                        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => {
                            serde_json::Value::Bool(b)
                        },
                        _ => continue,
                    };
                    metadata.insert(key, json_value);
                }
            }

            let point_id = if let Some(id) = scored_point.id {
                match id.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
                    None => continue,
                }
            } else {
                continue;
            };

            results.push(SearchResultItem {
                id: Uuid::parse_str(&point_id).unwrap_or_else(|_| Uuid::new_v4()),
                content,
                score: scored_point.score,
                metadata,
                document_id: metadata.get("document_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok()),
                chunk_index: metadata.get("chunk_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32),
            });
        }

        Ok(results)
    }

    async fn hybrid_search(
        &self,
        query_vector: &[f32],
        _query_text: Option<&str>,
        limit: u32,
        filter: Option<SearchFilter>,
    ) -> RagResult<Vec<SearchResultItem>> {
        // 简化实现：只使用向量搜索
        // 实际实现应该结合文本搜索和向量搜索结果
        self.search_vectors(query_vector, limit, filter).await
    }

    async fn get_collection_info(&self, collection_name: &str) -> RagResult<CollectionInfo> {
        let info = self.client
            .collection_info(collection_name)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("获取集合信息失败: {}", e)))?;

        let result = info.result.ok_or_else(|| {
            RagError::VectorStoreError("集合信息为空".to_string())
        })?;

        Ok(CollectionInfo {
            name: collection_name.to_string(),
            vectors_count: result.points_count.unwrap_or(0),
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            index_type: IndexType::HNSW, // Qdrant 默认使用 HNSW
            status: match result.status() {
                qdrant_client::qdrant::CollectionStatus::Green => CollectionStatus::Green,
                qdrant_client::qdrant::CollectionStatus::Yellow => CollectionStatus::Yellow,
                qdrant_client::qdrant::CollectionStatus::Red => CollectionStatus::Red,
            },
        })
    }

    async fn health_check(&self) -> RagResult<()> {
        self.client
            .health_check()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("健康检查失败: {}", e)))?;
        Ok(())
    }
}

/// 内存向量存储 (用于测试和小数据集)
pub struct InMemoryVectorStore {
    vectors: tokio::sync::RwLock<HashMap<String, VectorRecord>>,
    config: VectorStoreConfig,
}

impl InMemoryVectorStore {
    pub fn new(config: VectorStoreConfig) -> Self {
        Self {
            vectors: tokio::sync::RwLock::new(HashMap::new()),
            config,
        }
    }

    fn calculate_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Cosine => {
                crate::embedding::utils::cosine_similarity(a, b)
            },
            DistanceMetric::Euclidean => {
                1.0 / (1.0 + crate::embedding::utils::euclidean_distance(a, b))
            },
            DistanceMetric::Dot => {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            },
            DistanceMetric::Manhattan => {
                let distance: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                1.0 / (1.0 + distance)
            },
        }
    }

    fn matches_filter(&self, record: &VectorRecord, filter: &SearchFilter) -> bool {
        // 简化的过滤器实现
        for condition in &filter.must {
            if !self.matches_condition(record, condition) {
                return false;
            }
        }

        for condition in &filter.must_not {
            if self.matches_condition(record, condition) {
                return false;
            }
        }

        if !filter.should.is_empty() {
            let mut should_match = false;
            for condition in &filter.should {
                if self.matches_condition(record, condition) {
                    should_match = true;
                    break;
                }
            }
            if !should_match {
                return false;
            }
        }

        true
    }

    fn matches_condition(&self, record: &VectorRecord, condition: &FilterCondition) -> bool {
        let field_value = record.metadata.get(&condition.field);

        match condition.operator {
            FilterOperator::Equal => {
                field_value == Some(&condition.value)
            },
            FilterOperator::NotEqual => {
                field_value != Some(&condition.value)
            },
            FilterOperator::Contains => {
                if let (Some(field_val), Some(search_val)) = (
                    field_value.and_then(|v| v.as_str()),
                    condition.value.as_str()
                ) {
                    field_val.contains(search_val)
                } else {
                    false
                }
            },
            // 其他操作符的简化实现
            _ => false,
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn create_collection(&self, _config: &VectorStoreConfig) -> RagResult<()> {
        // 内存存储不需要创建集合
        Ok(())
    }

    async fn delete_collection(&self, _collection_name: &str) -> RagResult<()> {
        let mut vectors = self.vectors.write().await;
        vectors.clear();
        Ok(())
    }

    async fn collection_exists(&self, _collection_name: &str) -> RagResult<bool> {
        Ok(true)
    }

    async fn insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()> {
        let mut store = self.vectors.write().await;
        for vector in vectors {
            store.insert(vector.id.clone(), vector);
        }
        Ok(())
    }

    async fn batch_insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()> {
        self.insert_vectors(vectors).await
    }

    async fn update_vector(&self, id: &str, vector: VectorRecord) -> RagResult<()> {
        let mut store = self.vectors.write().await;
        store.insert(id.to_string(), vector);
        Ok(())
    }

    async fn delete_vector(&self, id: &str) -> RagResult<()> {
        let mut store = self.vectors.write().await;
        store.remove(id);
        Ok(())
    }

    async fn batch_delete_vectors(&self, ids: Vec<String>) -> RagResult<()> {
        let mut store = self.vectors.write().await;
        for id in ids {
            store.remove(&id);
        }
        Ok(())
    }

    async fn get_vector(&self, id: &str) -> RagResult<Option<VectorRecord>> {
        let store = self.vectors.read().await;
        Ok(store.get(id).cloned())
    }

    async fn search_vectors(
        &self,
        query_vector: &[f32],
        limit: u32,
        filter: Option<SearchFilter>,
    ) -> RagResult<Vec<SearchResultItem>> {
        let store = self.vectors.read().await;
        let mut results = Vec::new();

        for (id, record) in store.iter() {
            // 应用过滤器
            if let Some(ref filter) = filter {
                if !self.matches_filter(record, filter) {
                    continue;
                }
            }

            let score = self.calculate_similarity(query_vector, &record.vector);

            results.push(SearchResultItem {
                id: Uuid::parse_str(id).unwrap_or_else(|_| Uuid::new_v4()),
                content: record.text.clone().unwrap_or_default(),
                score,
                metadata: record.metadata.clone(),
                document_id: record.metadata.get("document_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok()),
                chunk_index: record.metadata.get("chunk_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32),
            });
        }

        // 按分数排序
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit as usize);

        Ok(results)
    }

    async fn hybrid_search(
        &self,
        query_vector: &[f32],
        _query_text: Option<&str>,
        limit: u32,
        filter: Option<SearchFilter>,
    ) -> RagResult<Vec<SearchResultItem>> {
        // 简化实现：只使用向量搜索
        self.search_vectors(query_vector, limit, filter).await
    }

    async fn get_collection_info(&self, collection_name: &str) -> RagResult<CollectionInfo> {
        let store = self.vectors.read().await;
        Ok(CollectionInfo {
            name: collection_name.to_string(),
            vectors_count: store.len() as u64,
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            index_type: IndexType::Flat,
            status: CollectionStatus::Green,
        })
    }

    async fn health_check(&self) -> RagResult<()> {
        Ok(())
    }
}

/// 向量存储工厂
pub struct VectorStoreFactory;

impl VectorStoreFactory {
    pub async fn create_store(config: VectorStoreConfig) -> RagResult<Box<dyn VectorStore>> {
        match config.provider {
            VectorStoreProvider::Qdrant => {
                let store = QdrantVectorStore::new(config).await?;
                Ok(Box::new(store))
            },
            VectorStoreProvider::InMemory => {
                let store = InMemoryVectorStore::new(config);
                Ok(Box::new(store))
            },
            _ => Err(RagError::ConfigurationError(format!(
                "不支持的向量存储提供商: {:?}", config.provider
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> VectorStoreConfig {
        VectorStoreConfig {
            provider: VectorStoreProvider::InMemory,
            connection_string: "memory://test".to_string(),
            collection_name: "test_collection".to_string(),
            dimensions: 128,
            distance_metric: DistanceMetric::Cosine,
            index_config: IndexConfig {
                index_type: IndexType::Flat,
                ef_construct: None,
                m: None,
                quantization: None,
            },
            batch_size: 100,
            timeout_seconds: 30,
        }
    }

    fn create_test_vector(id: &str, values: Vec<f32>) -> VectorRecord {
        VectorRecord {
            id: id.to_string(),
            vector: values,
            metadata: HashMap::from([
                ("test_field".to_string(), serde_json::Value::String("test_value".to_string())),
            ]),
            text: Some(format!("Test text for {}", id)),
        }
    }

    #[tokio::test]
    async fn test_inmemory_vector_store() {
        let config = create_test_config();
        let store = InMemoryVectorStore::new(config);

        // 测试创建集合
        assert!(store.create_collection(&create_test_config()).await.is_ok());

        // 测试插入向量
        let vectors = vec![
            create_test_vector("1", vec![1.0, 0.0, 0.0]),
            create_test_vector("2", vec![0.0, 1.0, 0.0]),
            create_test_vector("3", vec![0.0, 0.0, 1.0]),
        ];

        assert!(store.insert_vectors(vectors).await.is_ok());

        // 测试获取向量
        let retrieved = store.get_vector("1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "1");

        // 测试向量搜索
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search_vectors(&query, 2, None).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].score, 1.0); // 完全匹配

        // 测试删除向量
        assert!(store.delete_vector("1").await.is_ok());
        let retrieved = store.get_vector("1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_search_with_filter() {
        let config = create_test_config();
        let store = InMemoryVectorStore::new(config);

        let mut vector1 = create_test_vector("1", vec![1.0, 0.0]);
        vector1.metadata.insert("category".to_string(), serde_json::Value::String("A".to_string()));

        let mut vector2 = create_test_vector("2", vec![0.0, 1.0]);
        vector2.metadata.insert("category".to_string(), serde_json::Value::String("B".to_string()));

        store.insert_vectors(vec![vector1, vector2]).await.unwrap();

        let filter = SearchFilter {
            must: vec![FilterCondition {
                field: "category".to_string(),
                operator: FilterOperator::Equal,
                value: serde_json::Value::String("A".to_string()),
            }],
            must_not: vec![],
            should: vec![],
        };

        let results = store.search_vectors(&[1.0, 0.0], 10, Some(filter)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.get("category").unwrap().as_str().unwrap(), "A");
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let config = create_test_config();
        let store = InMemoryVectorStore::new(config);

        // 批量插入
        let vectors: Vec<VectorRecord> = (0..10)
            .map(|i| create_test_vector(&i.to_string(), vec![i as f32, 0.0]))
            .collect();

        assert!(store.batch_insert_vectors(vectors).await.is_ok());

        // 验证插入
        let info = store.get_collection_info("test").await.unwrap();
        assert_eq!(info.vectors_count, 10);

        // 批量删除
        let ids: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert!(store.batch_delete_vectors(ids).await.is_ok());

        let info = store.get_collection_info("test").await.unwrap();
        assert_eq!(info.vectors_count, 5);
    }

    #[test]
    fn test_distance_metrics() {
        let config = create_test_config();
        let store = InMemoryVectorStore::new(config);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        // 测试余弦相似度
        let similarity = store.calculate_similarity(&a, &b);
        assert_eq!(similarity, 0.0); // 垂直向量
    }

    #[test]
    fn test_vector_store_factory() {
        let config = create_test_config();

        // 测试工厂方法 (异步测试)
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let store = VectorStoreFactory::create_store(config).await.unwrap();
            assert!(store.health_check().await.is_ok());
        });
    }

    #[test]
    fn test_filter_serialization() {
        let filter = SearchFilter {
            must: vec![FilterCondition {
                field: "test".to_string(),
                operator: FilterOperator::Equal,
                value: serde_json::Value::String("value".to_string()),
            }],
            must_not: vec![],
            should: vec![],
        };

        let serialized = serde_json::to_string(&filter).unwrap();
        let deserialized: SearchFilter = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.must.len(), 1);
        assert_eq!(deserialized.must[0].field, "test");
    }
} create_collection(&self, config: &VectorStoreConfig) -> RagResult<()>;

/// 删除集合/索引
async fn delete_collection(&self, collection_name: &str) -> RagResult<()>;

/// 检查集合是否存在
async fn collection_exists(&self, collection_name: &str) -> RagResult<bool>;

/// 插入向量
async fn insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()>;

/// 批量插入向量
async fn batch_insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()>;

/// 更新向量
async fn update_vector(&self, id: &str, vector: VectorRecord) -> RagResult<()>;

/// 删除向量
async fn delete_vector(&self, id: &str) -> RagResult<()>;

/// 批量删除向量
async fn batch_delete_vectors(&self, ids: Vec<String>) -> RagResult<()>;

/// 获取向量
async fn get_vector(&self, id: &str) -> RagResult<Option<VectorRecord>>;

/// 向量搜索
async fn search_vectors(
    &self,
    query_vector: &[f32],
    limit: u32,
    filter: Option<SearchFilter>,
) -> RagResult<Vec<SearchResultItem>>;

/// 混合搜索 (向量 + 关键词)
async fn hybrid_search(
    &self,
    query_vector: &[f32],
    query_text: Option<&str>,
    limit: u32,
    filter: Option<SearchFilter>,
) -> RagResult<Vec<SearchResultItem>>;

/// 获取集合信息
async fn get_collection_info(&self, collection_name: &str) -> RagResult<CollectionInfo>;

/// 健康检查
async fn health_check(&self) -> RagResult<()>;
}

/// 向量记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub text: Option<String>,
}

/// 搜索过滤器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilter {
    pub must: Vec<FilterCondition>,
    pub must_not: Vec<FilterCondition>,
    pub should: Vec<FilterCondition>,
}

/// 过滤条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// 过滤操作符
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    In,
    NotIn,
    Contains,
    StartsWith,
    EndsWith,
}

/// 集合信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub vectors_count: u64,
    pub dimensions: u32,
    pub distance_metric: DistanceMetric,
    pub index_type: IndexType,
    pub status: CollectionStatus,
}

/// 集合状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CollectionStatus {
    Green,
    Yellow,
    Red,
}

/// Qdrant 向量存储实现
pub struct QdrantVectorStore {
    client: qdrant_client::client::QdrantClient,
    config: VectorStoreConfig,
}

impl QdrantVectorStore {
    pub async fn new(config: VectorStoreConfig) -> RagResult<Self> {
        let client = qdrant_client::client::QdrantClient::from_url(&config.connection_string)
            .build()
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant 连接失败: {}", e)))?;

        Ok(Self { client, config })
    }

    fn convert_distance_metric(&self, metric: &DistanceMetric) -> qdrant_client::qdrant::Distance {
        match metric {
            DistanceMetric::Cosine => qdrant_client::qdrant::Distance::Cosine,
            DistanceMetric::Euclidean => qdrant_client::qdrant::Distance::Euclid,
            DistanceMetric::Dot => qdrant_client::qdrant::Distance::Dot,
            DistanceMetric::Manhattan => qdrant_client::qdrant::Distance::Manhattan,
        }
    }

    fn convert_search_filter(&self, filter: &SearchFilter) -> qdrant_client::qdrant::Filter {
        use qdrant_client::qdrant::{Filter, Condition, FieldCondition, Match, Range};

        let mut conditions = Vec::new();

        // 处理 must 条件
        for condition in &filter.must {
            let field_condition = self.convert_filter_condition(condition);
            if let Some(fc) = field_condition {
                conditions.push(Condition {
                    condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(fc)),
                });
            }
        }

        Filter {
            should: vec![], // 简化实现
            must: conditions,
            must_not: vec![], // 简化实现
        }
    }

    fn convert_filter_condition(&self, condition: &FilterCondition) -> Option<FieldCondition> {
        use qdrant_client::qdrant::{FieldCondition, Match, Range, match_::MatchValue};

        match condition.operator {
            FilterOperator::Equal => {
                if let Some(string_val) = condition.value.as_str() {
                    Some(FieldCondition {
                        key: condition.field.clone(),
                        r#match: Some(Match {
                            match_value: Some(MatchValue::Keyword(string_val.to_string())),
                        }),
                        range: None,
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                    })
                } else if let Some(int_val) = condition.value.as_i64() {
                    Some(FieldCondition {
                        key: condition.field.clone(),
                        r#match: Some(Match {
                            match_value: Some(MatchValue::Integer(int_val)),
                        }),
                        range: None,
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                    })
                } else {
                    None
                }
            }
            FilterOperator::GreaterThan => {
                if let Some(float_val) = condition.value.as_f64() {
                    Some(FieldCondition {
                        key: condition.field.clone(),
                        r#match: None,
                        range: Some(Range {
                            lt: None,
                            gt: Some(float_val),
                            gte: None,
                            lte: None,
                        }),
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                    })
                } else {
                    None
                }
            }
            // 其他操作符的简化实现
            _ => None,
        }
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn create_collection(&self, config: &VectorStoreConfig) -> RagResult<()> {
        use qdrant_client::qdrant::{CreateCollection, VectorParams, VectorsConfig, vectors_config::Config};

        let vectors_config = VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: config.dimensions as u64,
                distance: self.convert_distance_metric(&config.distance_metric).into(),
                hnsw_config: None,
                quantization_config: None,
                on_disk: None,
            })),
        };

        let create_collection = CreateCollection {
            collection_name: config.collection_name.clone(),
            vectors_config: Some(vectors_config),
            shard_number: None,
            replication_factor: None,
            write_consistency_factor: None,
            on_disk_payload: None,
            timeout: Some(config.timeout_seconds as u64),
            optimizers_config: None,
            wal_config: None,
            quantization_config: None,
            init_from_collection: None,
        };

        self.client
            .create_collection(&create_collection)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("创建集合失败: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, collection_name: &str) -> RagResult<()> {
        use qdrant_client::qdrant::DeleteCollection;

        let delete_collection = DeleteCollection {
            collection_name: collection_name.to_string(),
            timeout: Some(self.config.timeout_seconds as u64),
        };

        self.client
            .delete_collection(&delete_collection)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("删除集合失败: {}", e)))?;

        Ok(())
    }

    async fn collection_exists(&self, collection_name: &str) -> RagResult<bool> {
        match self.client.collection_info(collection_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn insert_vectors(&self, vectors: Vec<VectorRecord>) -> RagResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        use qdrant_client::qdrant::{UpsertPoints, PointStruct, Value, Vectors, vectors::VectorsOptions};

        let points: Vec<PointStruct> = vectors
            .into_iter()
            .map(|record| {
                let mut payload = HashMap::new();

                // 添加文本内容
                if let Some(text) = record.text {
                    payload.insert("text".to_string(), Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(text)),
                    });
                }

                // 添加元数据
                for (key, value) in record.metadata {
                    let qdrant_value = match value {
                        serde_json::Value::String(s) => Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
                        },
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                Value {
                                    kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
                                }
                            } else if let Some(f) = n.as_f64() {
                                Value {
                                    kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)),
                                }
                            } else {
                                continue;
                            }
                        },
                        serde_json::Value::Bool(b) => Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(b)),
                        },
                        _ => continue,
                    };
                    payload.insert(key, qdrant_value);
                }

                PointStruct {
                    id: Some(qdrant_client::qdrant::PointId {
                        point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(record.id)),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(VectorsOptions::Vector(qdrant_client::qdrant::Vector {
                            data: record.vector,
                        })),
                    }),
                    payload,
                }
            })
            .collect();

        let upsert_points = UpsertPoints {
            collection_name: self.config.collection_name.clone(),
            wait: Some(true),
            points,
            ordering: None,
        };

        self.client
            .upsert_points(&upsert_points)
            .await
            .map_err(|e| RagError::VectorStoreError(format!("插入向量失败: {}", e)))?;

        Ok(())
    }

    async fn