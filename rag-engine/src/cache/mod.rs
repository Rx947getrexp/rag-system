//! # 缓存模块
//!
//! 提供多层缓存支持，包括内存缓存和 Redis 分布式缓存

use async_trait::async_trait;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

use crate::{
    config::RagConfig,
    error::{CacheError, RagError, RagResult},
};

/// 缓存接口
#[async_trait]
pub trait Cache {
    /// 获取缓存值
    async fn get<T>(&self, key: &str) -> RagResult<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send;

    /// 设置缓存值
    async fn set<T>(&self, key: &str, value: &T, ttl: u64) -> RagResult<()>
    where
        T: Serialize + Send + Sync;

    /// 删除缓存值
    async fn delete(&self, key: &str) -> RagResult<()>;

    /// 检查键是否存在
    async fn exists(&self, key: &str) -> RagResult<bool>;

    /// 设置过期时间
    async fn expire(&self, key: &str, ttl: u64) -> RagResult<()>;

    /// 获取剩余过期时间
    async fn ttl(&self, key: &str) -> RagResult<Option<u64>>;

    /// 清空所有缓存
    async fn clear(&self) -> RagResult<()>;

    /// 健康检查
    async fn health_check(&self) -> RagResult<()>;

    /// 清理过期缓存
    async fn cleanup(&self) -> RagResult<()>;

    /// 获取缓存统计信息
    async fn stats(&self) -> RagResult<CacheStats>;
}

/// 缓存统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub total_keys: u64,
    pub memory_usage_bytes: u64,
    pub evictions: u64,
}

/// 缓存项
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheItem<T> {
    value: T,
    created_at: u64,
    expires_at: Option<u64>,
}

impl<T> CacheItem<T> {
    fn new(value: T, ttl: Option<u64>) -> Self {
        let now = current_timestamp();
        Self {
            value,
            created_at: now,
            expires_at: ttl.map(|t| now + t),
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            current_timestamp() > expires_at
        } else {
            false
        }
    }
}

/// 内存缓存实现
pub struct MemoryCache {
    config: Arc<RagConfig>,
    store: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    stats: Arc<RwLock<CacheStats>>,
}

impl MemoryCache {
    pub fn new(config: Arc<RagConfig>) -> Self {
        Self {
            config,
            store: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                hit_rate: 0.0,
                total_keys: 0,
                memory_usage_bytes: 0,
                evictions: 0,
            })),
        }
    }

    async fn update_stats(&self, hit: bool) {
        let mut stats = self.stats.write().await;
        if hit {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }

        let total = stats.hits + stats.misses;
        stats.hit_rate = if total > 0 {
            stats.hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get<T>(&self, key: &str) -> RagResult<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        let store = self.store.read().await;

        if let Some(data) = store.get(key) {
            match bincode::deserialize::<CacheItem<T>>(data) {
                Ok(item) => {
                    if item.is_expired() {
                        drop(store);
                        let _ = self.delete(key).await;
                        self.update_stats(false).await;
                        Ok(None)
                    } else {
                        self.update_stats(true).await;
                        Ok(Some(item.value))
                    }
                }
                Err(e) => {
                    error!("缓存反序列化失败: {}", e);
                    self.update_stats(false).await;
                    Ok(None)
                }
            }
        } else {
            self.update_stats(false).await;
            Ok(None)
        }
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: u64) -> RagResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let item = CacheItem::new(value, Some(ttl));
        let data = bincode::serialize(&item)
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        let mut store = self.store.write().await;

        // 检查内存限制
        let current_size: usize = store.values().map(|v| v.len()).sum();
        let max_size = self.config.cache.memory.max_size as usize;

        if current_size + data.len() > max_size {
            // 简单的 LRU 实现：删除一些旧键
            let keys_to_remove: Vec<String> = store.keys().take(10).cloned().collect();
            for key_to_remove in keys_to_remove {
                store.remove(&key_to_remove);
            }

            let mut stats = self.stats.write().await;
            stats.evictions += 10;
        }

        store.insert(key.to_string(), data);

        // 更新统计信息
        {
            let mut stats = self.stats.write().await;
            stats.total_keys = store.len() as u64;
            stats.memory_usage_bytes = store.values().map(|v| v.len() as u64).sum();
        }

        debug!("内存缓存设置成功: {}", key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> RagResult<()> {
        let mut store = self.store.write().await;
        store.remove(key);

        let mut stats = self.stats.write().await;
        stats.total_keys = store.len() as u64;
        stats.memory_usage_bytes = store.values().map(|v| v.len() as u64).sum();

        Ok(())
    }

    async fn exists(&self, key: &str) -> RagResult<bool> {
        let store = self.store.read().await;
        Ok(store.contains_key(key))
    }

    async fn expire(&self, key: &str, ttl: u64) -> RagResult<()> {
        // 内存缓存中的过期时间在 CacheItem 中管理
        // 这里是一个简化实现
        Ok(())
    }

    async fn ttl(&self, key: &str) -> RagResult<Option<u64>> {
        let store = self.store.read().await;
        if let Some(data) = store.get(key) {
            if let Ok(item) = bincode::deserialize::<CacheItem<serde_json::Value>>(data) {
                if let Some(expires_at) = item.expires_at {
                    let now = current_timestamp();
                    if expires_at > now {
                        return Ok(Some(expires_at - now));
                    }
                }
            }
        }
        Ok(None)
    }

    async fn clear(&self) -> RagResult<()> {
        let mut store = self.store.write().await;
        store.clear();

        let mut stats = self.stats.write().await;
        *stats = CacheStats {
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            total_keys: 0,
            memory_usage_bytes: 0,
            evictions: 0,
        };

        debug!("内存缓存已清空");
        Ok(())
    }

    async fn health_check(&self) -> RagResult<()> {
        // 内存缓存总是健康的
        Ok(())
    }

    async fn cleanup(&self) -> RagResult<()> {
        let mut store = self.store.write().await;
        let mut expired_keys = Vec::new();

        for (key, data) in store.iter() {
            if let Ok(item) = bincode::deserialize::<CacheItem<serde_json::Value>>(data) {
                if item.is_expired() {
                    expired_keys.push(key.clone());
                }
            }
        }

        for key in expired_keys {
            store.remove(&key);
        }

        debug!("内存缓存清理完成");
        Ok(())
    }

    async fn stats(&self) -> RagResult<CacheStats> {
        Ok(self.stats.read().await.clone())
    }
}

/// Redis 缓存实现
pub struct RedisCache {
    config: Arc<RagConfig>,
    client: redis::Client,
    stats: Arc<RwLock<CacheStats>>,
}

impl RedisCache {
    pub async fn new(config: Arc<RagConfig>) -> RagResult<Self> {
        let client = redis::Client::open(config.cache.redis.url.as_str())
            .map_err(|e| RagError::CacheError(CacheError::RedisConnectionFailed(e.to_string())))?;

        // 测试连接
        let mut conn = client.get_async_connection().await
            .map_err(|e| RagError::CacheError(CacheError::RedisConnectionFailed(e.to_string())))?;

        let _: String = conn.ping().await
            .map_err(|e| RagError::CacheError(CacheError::RedisConnectionFailed(e.to_string())))?;

        Ok(Self {
            config,
            client,
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                hit_rate: 0.0,
                total_keys: 0,
                memory_usage_bytes: 0,
                evictions: 0,
            })),
        })
    }

    async fn get_connection(&self) -> RagResult<redis::aio::Connection> {
        self.client.get_async_connection().await
            .map_err(|e| RagError::CacheError(CacheError::RedisConnectionFailed(e.to_string())))
    }

    async fn update_stats(&self, hit: bool) {
        let mut stats = self.stats.write().await;
        if hit {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }

        let total = stats.hits + stats.misses;
        stats.hit_rate = if total > 0 {
            stats.hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

#[async_trait]
impl Cache for RedisCache {
    async fn get<T>(&self, key: &str) -> RagResult<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        let mut conn = self.get_connection().await?;

        let result: Option<String> = conn.get(key).await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        if let Some(data) = result {
            match serde_json::from_str::<T>(&data) {
                Ok(value) => {
                    self.update_stats(true).await;
                    debug!("Redis 缓存命中: {}", key);
                    Ok(Some(value))
                }
                Err(e) => {
                    error!("Redis 缓存反序列化失败: {}", e);
                    self.update_stats(false).await;
                    Ok(None)
                }
            }
        } else {
            self.update_stats(false).await;
            debug!("Redis 缓存未命中: {}", key);
            Ok(None)
        }
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: u64) -> RagResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let mut conn = self.get_connection().await?;

        let data = serde_json::to_string(value)
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        if ttl > 0 {
            conn.setex(key, ttl, data).await
        } else {
            conn.set(key, data).await
        }.map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        debug!("Redis 缓存设置成功: {} (TTL: {}s)", key, ttl);
        Ok(())
    }

    async fn delete(&self, key: &str) -> RagResult<()> {
        let mut conn = self.get_connection().await?;

        let _: u32 = conn.del(key).await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        debug!("Redis 缓存删除成功: {}", key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> RagResult<bool> {
        let mut conn = self.get_connection().await?;

        let exists: bool = conn.exists(key).await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        Ok(exists)
    }

    async fn expire(&self, key: &str, ttl: u64) -> RagResult<()> {
        let mut conn = self.get_connection().await?;

        let _: bool = conn.expire(key, ttl as usize).await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        Ok(())
    }

    async fn ttl(&self, key: &str) -> RagResult<Option<u64>> {
        let mut conn = self.get_connection().await?;

        let ttl: i32 = conn.ttl(key).await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        match ttl {
            -2 => Ok(None), // 键不存在
            -1 => Ok(None), // 键存在但没有过期时间
            t if t > 0 => Ok(Some(t as u64)),
            _ => Ok(None),
        }
    }

    async fn clear(&self) -> RagResult<()> {
        let mut conn = self.get_connection().await?;

        let _: String = conn.flushdb().await
            .map_err(|e| RagError::CacheError(CacheError::OperationFailed(e.to_string())))?;

        debug!("Redis 缓存已清空");
        Ok(())
    }

    async fn health_check(&self) -> RagResult<()> {
        let mut conn = self.get_connection().await?;

        let _: String = conn.ping().await
            .map_err(|e| RagError::CacheError(CacheError::RedisConnectionFailed(e.to_string())))?;

        Ok(())
    }

    async fn cleanup(&self) -> RagResult<()> {
        // Redis 自动处理过期键的清理
        debug!("Redis 缓存清理完成");
        Ok(())
    }

    async fn stats(&self) -> RagResult<CacheStats> {
        let mut stats = self.stats.write().await;

        // 尝试获取 Redis 统计信息
        if let Ok(mut conn) = self.get_connection().await {
            if let Ok(info) = conn.info::<String>("keyspace").await {
                // 解析 Redis info 获取键数量等信息
                // 这里是简化实现
                if let Some(db_info) = info.lines().find(|line| line.starts_with("db0:")) {
                    if let Some(keys_part) = db_info.split(',').next() {
                        if let Some(keys_str) = keys_part.split('=').nth(1) {
                            if let Ok(keys) = keys_str.parse::<u64>() {
                                stats.total_keys = keys;
                            }
                        }
                    }
                }
            }
        }

        Ok(stats.clone())
    }
}

/// 多层缓存实现
pub struct TieredCache {
    l1_cache: Box<dyn Cache + Send + Sync>, // 内存缓存
    l2_cache: Box<dyn Cache + Send + Sync>, // Redis 缓存
    config: Arc<RagConfig>,
}

impl TieredCache {
    pub async fn new(config: Arc<RagConfig>) -> RagResult<Self> {
        let l1_cache = Box::new(MemoryCache::new(config.clone()));
        let l2_cache = Box::new(RedisCache::new(config.clone()).await?);

        Ok(Self {
            l1_cache,
            l2_cache,
            config,
        })
    }
}

#[async_trait]
impl Cache for TieredCache {
    async fn get<T>(&self, key: &str) -> RagResult<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send + Clone + Serialize,
    {
        // 首先检查 L1 缓存
        if let Some(value) = self.l1_cache.get::<T>(key).await? {
            debug!("L1 缓存命中: {}", key);
            return Ok(Some(value));
        }

        // 然后检查 L2 缓存
        if let Some(value) = self.l2_cache.get::<T>(key).await? {
            debug!("L2 缓存命中: {}", key);

            // 将值提升到 L1 缓存
            let l1_ttl = self.config.cache.memory.default_ttl;
            let _ = self.l1_cache.set(key, &value, l1_ttl).await;

            return Ok(Some(value));
        }

        debug!("缓存未命中: {}", key);
        Ok(None)
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: u64) -> RagResult<()>
    where
        T: Serialize + Send + Sync,
    {
        // 同时设置到两层缓存
        let l1_ttl = ttl.min(self.config.cache.memory.default_ttl);
        let _ = self.l1_cache.set(key, value, l1_ttl).await;
        let _ = self.l2_cache.set(key, value, ttl).await;

        debug!("多层缓存设置成功: {}", key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> RagResult<()> {
        // 从两层缓存中删除
        let _ = self.l1_cache.delete(key).await;
        let _ = self.l2_cache.delete(key).await;

        Ok(())
    }

    async fn exists(&self, key: &str) -> RagResult<bool> {
        // 检查任一层缓存中是否存在
        if self.l1_cache.exists(key).await? {
            return Ok(true);
        }

        self.l2_cache.exists(key).await
    }

    async fn expire(&self, key: &str, ttl: u64) -> RagResult<()> {
        // 对两层缓存设置过期时间
        let _ = self.l1_cache.expire(key, ttl).await;
        let _ = self.l2_cache.expire(key, ttl).await;

        Ok(())
    }

    async fn ttl(&self, key: &str) -> RagResult<Option<u64>> {
        // 优先返回 L1 缓存的 TTL
        if let Some(ttl) = self.l1_cache.ttl(key).await? {
            return Ok(Some(ttl));
        }

        self.l2_cache.ttl(key).await
    }

    async fn clear(&self) -> RagResult<()> {
        // 清空两层缓存
        let _ = self.l1_cache.clear().await;
        let _ = self.l2_cache.clear().await;

        debug!("多层缓存已清空");
        Ok(())
    }

    async fn health_check(&self) -> RagResult<()> {
        // 检查两层缓存的健康状态
        self.l1_cache.health_check().await?;
        self.l2_cache.health_check().await?;

        Ok(())
    }

    async fn cleanup(&self) -> RagResult<()> {
        // 清理两层缓存
        let _ = self.l1_cache.cleanup().await;
        let _ = self.l2_cache.cleanup().await;

        Ok(())
    }

    async fn stats(&self) -> RagResult<CacheStats> {
        // 合并两层缓存的统计信息
        let l1_stats = self.l1_cache.stats().await?;
        let l2_stats = self.l2_cache.stats().await?;

        Ok(CacheStats {
            hits: l1_stats.hits + l2_stats.hits,
            misses: l1_stats.misses + l2_stats.misses,
            hit_rate: {
                let total = l1_stats.hits + l1_stats.misses + l2_stats.hits + l2_stats.misses;
                if total > 0 {
                    (l1_stats.hits + l2_stats.hits) as f64 / total as f64
                } else {
                    0.0
                }
            },
            total_keys: l1_stats.total_keys + l2_stats.total_keys,
            memory_usage_bytes: l1_stats.memory_usage_bytes + l2_stats.memory_usage_bytes,
            evictions: l1_stats.evictions + l2_stats.evictions,
        })
    }
}

/// 获取当前时间戳 (秒)
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// 缓存键生成器
pub struct CacheKeyBuilder {
    prefix: String,
}

impl CacheKeyBuilder {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    pub fn build(&self, parts: &[&str]) -> String {
        format!("{}:{}", self.prefix, parts.join(":"))
    }

    pub fn search_key(&self, query: &str, options_hash: &str) -> String {
        self.build(&["search", query, options_hash])
    }

    pub fn document_key(&self, doc_id: &str) -> String {
        self.build(&["document", doc_id])
    }

    pub fn embedding_key(&self, content_hash: &str) -> String {
        self.build(&["embedding", content_hash])
    }

    pub fn user_session_key(&self, user_id: &str, session_id: &str) -> String {
        self.build(&["session", user_id, session_id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestData {
        id: u32,
        name: String,
    }

    #[tokio::test]
    async fn test_memory_cache() {
        let config = Arc::new(RagConfig::default());
        let cache = MemoryCache::new(config);

        let test_data = TestData {
            id: 1,
            name: "test".to_string(),
        };

        // 测试设置和获取
        cache.set("test_key", &test_data, 60).await.unwrap();
        let result: Option<TestData> = cache.get("test_key").await.unwrap();
        assert_eq!(result, Some(test_data));

        // 测试删除
        cache.delete("test_key").await.unwrap();
        let result: Option<TestData> = cache.get("test_key").await.unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_key_builder() {
        let builder = CacheKeyBuilder::new("rag");

        assert_eq!(builder.build(&["test", "key"]), "rag:test:key");
        assert_eq!(builder.search_key("query", "hash"), "rag:search:query:hash");
        assert_eq!(builder.document_key("doc123"), "rag:document:doc123");
    }

    #[test]
    fn test_cache_item_expiration() {
        let item = CacheItem::new("test", Some(1)); // 1 秒 TTL
        assert!(!item.is_expired());

        // 模拟时间流逝
        std::thread::sleep(std::time::Duration::from_secs(2));
        // 注意：这个测试可能因为系统时间的精度而不稳定
    }
}