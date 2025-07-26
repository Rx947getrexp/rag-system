//! # 文档处理模块
//!
//! 处理各种格式的文档，包括文本提取、分块、清理等
//! 文件路径: rag-engine/src/pipeline/document_processor.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::io::AsyncReadExt;
use uuid::Uuid;

use crate::error::{RagError, RagResult};
use crate::types::*;

/// 支持的文档格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Pdf,
    Docx,
    Rtf,
    Json,
    Csv,
}

/// 文档元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
    pub language: Option<String>,
    pub page_count: Option<u32>,
    pub word_count: Option<u32>,
    pub char_count: u32,
    pub file_size: u64,
    pub mime_type: String,
    pub custom_fields: HashMap<String, String>,
}

/// 处理后的文档
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub id: Uuid,
    pub original_filename: String,
    pub format: DocumentFormat,
    pub content: String,
    pub metadata: DocumentMetadata,
    pub chunks: Vec<DocumentChunk>,
    pub processing_stats: ProcessingStats,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// 文档块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub content: String,
    pub chunk_index: u32,
    pub start_char: u32,
    pub end_char: u32,
    pub metadata: ChunkMetadata,
}

/// 块元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub section_title: Option<String>,
    pub page_number: Option<u32>,
    pub paragraph_index: Option<u32>,
    pub semantic_type: Option<String>, // heading, paragraph, list, table, etc.
    pub confidence_score: f32,
}

/// 处理统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub processing_time_ms: u64,
    pub total_chars: u32,
    pub total_words: u32,
    pub chunks_created: u32,
    pub errors_count: u32,
    pub warnings_count: u32,
}

/// 分块配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub strategy: ChunkingStrategy,
    pub max_chunk_size: u32,
    pub chunk_overlap: u32,
    pub min_chunk_size: u32,
    pub respect_sentence_boundaries: bool,
    pub respect_paragraph_boundaries: bool,
    pub preserve_structure: bool,
}

/// 分块策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    FixedSize,
    Sentence,
    Paragraph,
    Semantic,
    Hybrid,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Hybrid,
            max_chunk_size: 1000,
            chunk_overlap: 200,
            min_chunk_size: 100,
            respect_sentence_boundaries: true,
            respect_paragraph_boundaries: true,
            preserve_structure: true,
        }
    }
}

/// 文档处理器 trait
#[async_trait]
pub trait DocumentProcessor: Send + Sync {
    /// 检查是否支持该格式
    fn supports_format(&self, format: &DocumentFormat) -> bool;

    /// 从字节数据提取文本
    async fn extract_text(&self, data: &[u8], format: &DocumentFormat) -> RagResult<String>;

    /// 提取文档元数据
    async fn extract_metadata(&self, data: &[u8], format: &DocumentFormat) -> RagResult<DocumentMetadata>;

    /// 检测文档格式
    fn detect_format(&self, data: &[u8], filename: Option<&str>) -> RagResult<DocumentFormat>;
}

/// 主文档处理器
pub struct MainDocumentProcessor {
    text_processor: TextProcessor,
    markdown_processor: MarkdownProcessor,
    html_processor: HtmlProcessor,
    pdf_processor: PdfProcessor,
    docx_processor: DocxProcessor,
    json_processor: JsonProcessor,
    csv_processor: CsvProcessor,
    chunking_config: ChunkingConfig,
}

impl MainDocumentProcessor {
    pub fn new(chunking_config: ChunkingConfig) -> Self {
        Self {
            text_processor: TextProcessor::new(),
            markdown_processor: MarkdownProcessor::new(),
            html_processor: HtmlProcessor::new(),
            pdf_processor: PdfProcessor::new(),
            docx_processor: DocxProcessor::new(),
            json_processor: JsonProcessor::new(),
            csv_processor: CsvProcessor::new(),
            chunking_config,
        }
    }

    /// 处理文档
    pub async fn process_document(
        &self,
        data: Vec<u8>,
        filename: String,
        custom_metadata: Option<HashMap<String, String>>,
    ) -> RagResult<ProcessedDocument> {
        let start_time = std::time::Instant::now();
        let format = self.detect_format(&data, Some(&filename))?;

        tracing::info!("开始处理文档: {} (格式: {:?})", filename, format);

        // 提取文本内容
        let content = self.extract_text(&data, &format).await?;

        // 提取元数据
        let mut metadata = self.extract_metadata(&data, &format).await?;
        metadata.file_size = data.len() as u64;
        metadata.char_count = content.chars().count() as u32;

        // 合并自定义元数据
        if let Some(custom) = custom_metadata {
            metadata.custom_fields.extend(custom);
        }

        // 生成文档块
        let chunks = self.chunk_document(&content, &format).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let processed_doc = ProcessedDocument {
            id: Uuid::new_v4(),
            original_filename: filename,
            format,
            content,
            metadata,
            chunks: chunks.clone(),
            processing_stats: ProcessingStats {
                processing_time_ms: processing_time,
                total_chars: content.chars().count() as u32,
                total_words: content.split_whitespace().count() as u32,
                chunks_created: chunks.len() as u32,
                errors_count: 0,
                warnings_count: 0,
            },
            created_at: chrono::Utc::now(),
        };

        tracing::info!(
            "文档处理完成: {} 字符, {} 个块, 耗时 {}ms",
            processed_doc.processing_stats.total_chars,
            processed_doc.processing_stats.chunks_created,
            processing_time
        );

        Ok(processed_doc)
    }

    /// 将文档分块
    async fn chunk_document(
        &self,
        content: &str,
        format: &DocumentFormat,
    ) -> RagResult<Vec<DocumentChunk>> {
        let chunker = DocumentChunker::new(self.chunking_config.clone());
        chunker.chunk_text(content, format).await
    }

    /// 获取适当的处理器
    fn get_processor(&self, format: &DocumentFormat) -> &dyn DocumentProcessor {
        match format {
            DocumentFormat::PlainText => &self.text_processor,
            DocumentFormat::Markdown => &self.markdown_processor,
            DocumentFormat::Html => &self.html_processor,
            DocumentFormat::Pdf => &self.pdf_processor,
            DocumentFormat::Docx => &self.docx_processor,
            DocumentFormat::Json => &self.json_processor,
            DocumentFormat::Csv => &self.csv_processor,
            DocumentFormat::Rtf => &self.text_processor, // fallback
        }
    }
}

#[async_trait]
impl DocumentProcessor for MainDocumentProcessor {
    fn supports_format(&self, format: &DocumentFormat) -> bool {
        matches!(
            format,
            DocumentFormat::PlainText
                | DocumentFormat::Markdown
                | DocumentFormat::Html
                | DocumentFormat::Pdf
                | DocumentFormat::Docx
                | DocumentFormat::Json
                | DocumentFormat::Csv
                | DocumentFormat::Rtf
        )
    }

    async fn extract_text(&self, data: &[u8], format: &DocumentFormat) -> RagResult<String> {
        let processor = self.get_processor(format);
        processor.extract_text(data, format).await
    }

    async fn extract_metadata(&self, data: &[u8], format: &DocumentFormat) -> RagResult<DocumentMetadata> {
        let processor = self.get_processor(format);
        processor.extract_metadata(data, format).await
    }

    fn detect_format(&self, data: &[u8], filename: Option<&str>) -> RagResult<DocumentFormat> {
        // 首先根据文件扩展名检测
        if let Some(filename) = filename {
            let extension = Path::new(filename)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase());

            if let Some(ext) = extension {
                match ext.as_str() {
                    "txt" => return Ok(DocumentFormat::PlainText),
                    "md" | "markdown" => return Ok(DocumentFormat::Markdown),
                    "html" | "htm" => return Ok(DocumentFormat::Html),
                    "pdf" => return Ok(DocumentFormat::Pdf),
                    "docx" | "doc" => return Ok(DocumentFormat::Docx),
                    "rtf" => return Ok(DocumentFormat::Rtf),
                    "json" => return Ok(DocumentFormat::Json),
                    "csv" => return Ok(DocumentFormat::Csv),
                    _ => {}
                }
            }
        }

        // 根据内容特征检测
        if data.len() >= 4 {
            // PDF 文件头
            if &data[0..4] == b"%PDF" {
                return Ok(DocumentFormat::Pdf);
            }

            // ZIP 格式 (DOCX)
            if &data[0..4] == b"PK\x03\x04" {
                return Ok(DocumentFormat::Docx);
            }
        }

        // JSON 检测
        if let Ok(text) = std::str::from_utf8(data) {
            let trimmed = text.trim();
            if (trimmed.starts_with('{') && trimmed.ends_with('}'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
            {
                return Ok(DocumentFormat::Json);
            }

            // HTML 检测
            if trimmed.starts_with("<!DOCTYPE html")
                || trimmed.starts_with("<html")
                || trimmed.contains("<body")
            {
                return Ok(DocumentFormat::Html);
            }

            // Markdown 检测 (简单启发式)
            if trimmed.contains("# ") || trimmed.contains("## ") || trimmed.contains("```") {
                return Ok(DocumentFormat::Markdown);
            }
        }

        // 默认为纯文本
        Ok(DocumentFormat::PlainText)
    }
}

/// 纯文本处理器
pub struct TextProcessor;

impl TextProcessor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentProcessor for TextProcessor {
    fn supports_format(&self, format: &DocumentFormat) -> bool {
        matches!(format, DocumentFormat::PlainText)
    }

    async fn extract_text(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<String> {
        String::from_utf8(data.to_vec())
            .map_err(|e| RagError::DocumentProcessingError(format!("UTF-8 解码失败: {}", e)))
    }

    async fn extract_metadata(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<DocumentMetadata> {
        let content = self.extract_text(data, &DocumentFormat::PlainText).await?;

        Ok(DocumentMetadata {
            title: None,
            author: None,
            created_at: None,
            modified_at: None,
            language: detect_language(&content),
            page_count: None,
            word_count: Some(content.split_whitespace().count() as u32),
            char_count: content.chars().count() as u32,
            file_size: data.len() as u64,
            mime_type: "text/plain".to_string(),
            custom_fields: HashMap::new(),
        })
    }

    fn detect_format(&self, _data: &[u8], _filename: Option<&str>) -> RagResult<DocumentFormat> {
        Ok(DocumentFormat::PlainText)
    }
}

/// Markdown 处理器
pub struct MarkdownProcessor;

impl MarkdownProcessor {
    pub fn new() -> Self {
        Self
    }

    /// 提取 Markdown 标题
    fn extract_headings(&self, content: &str) -> Vec<String> {
        content
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with('#') {
                    Some(trimmed.trim_start_matches('#').trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[async_trait]
impl DocumentProcessor for MarkdownProcessor {
    fn supports_format(&self, format: &DocumentFormat) -> bool {
        matches!(format, DocumentFormat::Markdown)
    }

    async fn extract_text(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<String> {
        let content = String::from_utf8(data.to_vec())
            .map_err(|e| RagError::DocumentProcessingError(format!("UTF-8 解码失败: {}", e)))?;

        // 基本的 Markdown 清理 (移除格式标记)
        let cleaned = content
            .lines()
            .map(|line| {
                let mut cleaned_line = line.to_string();

                // 移除标题标记
                if cleaned_line.trim().starts_with('#') {
                    cleaned_line = cleaned_line.trim_start_matches('#').trim().to_string();
                }

                // 移除粗体和斜体标记
                cleaned_line = cleaned_line.replace("**", "").replace("*", "");

                // 移除代码块标记
                if cleaned_line.trim().starts_with("```") {
                    return String::new();
                }

                cleaned_line
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(cleaned)
    }

    async fn extract_metadata(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<DocumentMetadata> {
        let content = String::from_utf8(data.to_vec())
            .map_err(|e| RagError::DocumentProcessingError(format!("UTF-8 解码失败: {}", e)))?;

        let headings = self.extract_headings(&content);
        let title = headings.first().cloned();

        Ok(DocumentMetadata {
            title,
            author: None,
            created_at: None,
            modified_at: None,
            language: detect_language(&content),
            page_count: None,
            word_count: Some(content.split_whitespace().count() as u32),
            char_count: content.chars().count() as u32,
            file_size: data.len() as u64,
            mime_type: "text/markdown".to_string(),
            custom_fields: HashMap::new(),
        })
    }

    fn detect_format(&self, _data: &[u8], _filename: Option<&str>) -> RagResult<DocumentFormat> {
        Ok(DocumentFormat::Markdown)
    }
}

// 其他处理器的简化实现
pub struct HtmlProcessor;
pub struct PdfProcessor;
pub struct DocxProcessor;
pub struct JsonProcessor;
pub struct CsvProcessor;

impl HtmlProcessor {
    pub fn new() -> Self { Self }
}

impl PdfProcessor {
    pub fn new() -> Self { Self }
}

impl DocxProcessor {
    pub fn new() -> Self { Self }
}

impl JsonProcessor {
    pub fn new() -> Self { Self }
}

impl CsvProcessor {
    pub fn new() -> Self { Self }
}

// 为所有处理器实现基本的 DocumentProcessor trait
macro_rules! impl_basic_processor {
    ($processor:ty, $format:expr, $mime_type:expr) => {
        #[async_trait]
        impl DocumentProcessor for $processor {
            fn supports_format(&self, format: &DocumentFormat) -> bool {
                *format == $format
            }

            async fn extract_text(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<String> {
                // 简化实现 - 实际项目中需要使用专门的库
                String::from_utf8(data.to_vec())
                    .map_err(|e| RagError::DocumentProcessingError(format!("文本提取失败: {}", e)))
            }

            async fn extract_metadata(&self, data: &[u8], _format: &DocumentFormat) -> RagResult<DocumentMetadata> {
                Ok(DocumentMetadata {
                    title: None,
                    author: None,
                    created_at: None,
                    modified_at: None,
                    language: None,
                    page_count: None,
                    word_count: None,
                    char_count: 0,
                    file_size: data.len() as u64,
                    mime_type: $mime_type.to_string(),
                    custom_fields: HashMap::new(),
                })
            }

            fn detect_format(&self, _data: &[u8], _filename: Option<&str>) -> RagResult<DocumentFormat> {
                Ok($format)
            }
        }
    };
}

impl_basic_processor!(HtmlProcessor, DocumentFormat::Html, "text/html");
impl_basic_processor!(PdfProcessor, DocumentFormat::Pdf, "application/pdf");
impl_basic_processor!(DocxProcessor, DocumentFormat::Docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
impl_basic_processor!(JsonProcessor, DocumentFormat::Json, "application/json");
impl_basic_processor!(CsvProcessor, DocumentFormat::Csv, "text/csv");

/// 文档分块器
pub struct DocumentChunker {
    config: ChunkingConfig,
}

impl DocumentChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    pub async fn chunk_text(
        &self,
        text: &str,
        _format: &DocumentFormat,
    ) -> RagResult<Vec<DocumentChunk>> {
        match self.config.strategy {
            ChunkingStrategy::FixedSize => self.chunk_by_fixed_size(text).await,
            ChunkingStrategy::Sentence => self.chunk_by_sentence(text).await,
            ChunkingStrategy::Paragraph => self.chunk_by_paragraph(text).await,
            ChunkingStrategy::Semantic => self.chunk_by_semantic(text).await,
            ChunkingStrategy::Hybrid => self.chunk_hybrid(text).await,
        }
    }

    async fn chunk_by_fixed_size(&self, text: &str) -> RagResult<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < chars.len() {
            let end = std::cmp::min(start + self.config.max_chunk_size as usize, chars.len());
            let chunk_text: String = chars[start..end].iter().collect();

            let chunk = DocumentChunk {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(), // Will be set by caller
                content: chunk_text,
                chunk_index,
                start_char: start as u32,
                end_char: end as u32,
                metadata: ChunkMetadata {
                    section_title: None,
                    page_number: None,
                    paragraph_index: None,
                    semantic_type: Some("fixed_size".to_string()),
                    confidence_score: 1.0,
                },
            };

            chunks.push(chunk);
            chunk_index += 1;

            // 计算下一个块的起始位置 (考虑重叠)
            start = if end == chars.len() {
                end
            } else {
                std::cmp::max(
                    start + self.config.max_chunk_size as usize - self.config.chunk_overlap as usize,
                    start + 1,
                )
            };
        }

        Ok(chunks)
    }

    async fn chunk_by_sentence(&self, text: &str) -> RagResult<Vec<DocumentChunk>> {
        // 简化的句子分割
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_index = 0;
        let mut start_char = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if current_chunk.len() + sentence.len() > self.config.max_chunk_size as usize {
                if !current_chunk.is_empty() {
                    let chunk = DocumentChunk {
                        id: Uuid::new_v4(),
                        document_id: Uuid::new_v4(),
                        content: current_chunk.trim().to_string(),
                        chunk_index,
                        start_char,
                        end_char: start_char + current_chunk.len() as u32,
                        metadata: ChunkMetadata {
                            section_title: None,
                            page_number: None,
                            paragraph_index: None,
                            semantic_type: Some("sentence".to_string()),
                            confidence_score: 0.9,
                        },
                    };
                    chunks.push(chunk);
                    chunk_index += 1;
                    start_char += current_chunk.len() as u32;
                }
                current_chunk = sentence.to_string();
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }

        // 添加最后一个块
        if !current_chunk.is_empty() {
            let chunk = DocumentChunk {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                content: current_chunk.trim().to_string(),
                chunk_index,
                start_char,
                end_char: start_char + current_chunk.len() as u32,
                metadata: ChunkMetadata {
                    section_title: None,
                    page_number: None,
                    paragraph_index: None,
                    semantic_type: Some("sentence".to_string()),
                    confidence_score: 0.9,
                },
            };
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    async fn chunk_by_paragraph(&self, text: &str) -> RagResult<Vec<DocumentChunk>> {
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut char_position = 0;

        for (para_index, paragraph) in paragraphs.iter().enumerate() {
            let paragraph = paragraph.trim();

            if paragraph.len() <= self.config.max_chunk_size as usize {
                // 段落足够小，直接作为一个块
                let chunk = DocumentChunk {
                    id: Uuid::new_v4(),
                    document_id: Uuid::new_v4(),
                    content: paragraph.to_string(),
                    chunk_index,
                    start_char: char_position,
                    end_char: char_position + paragraph.len() as u32,
                    metadata: ChunkMetadata {
                        section_title: None,
                        page_number: None,
                        paragraph_index: Some(para_index as u32),
                        semantic_type: Some("paragraph".to_string()),
                        confidence_score: 0.95,
                    },
                };
                chunks.push(chunk);
                chunk_index += 1;
            } else {
                // 段落太大，需要进一步分割
                let sub_chunks = self.chunk_by_fixed_size(paragraph).await?;
                for mut sub_chunk in sub_chunks {
                    sub_chunk.chunk_index = chunk_index;
                    sub_chunk.start_char += char_position;
                    sub_chunk.end_char += char_position;
                    sub_chunk.metadata.paragraph_index = Some(para_index as u32);
                    chunks.push(sub_chunk);
                    chunk_index += 1;
                }
            }

            char_position += paragraph.len() as u32 + 2; // +2 for \n\n
        }

        Ok(chunks)
    }

    async fn chunk_by_semantic(&self, text: &str) -> RagResult<Vec<DocumentChunk>> {
        // 简化的语义分块 - 实际实现需要 NLP 模型
        self.chunk_by_paragraph(text).await
    }

    async fn chunk_hybrid(&self, text: &str) -> RagResult<Vec<DocumentChunk>> {
        // 混合策略：优先按段落，超大段落按句子分割
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut char_position = 0;

        for (para_index, paragraph) in paragraphs.iter().enumerate() {
            let paragraph = paragraph.trim();

            if paragraph.len() <= self.config.max_chunk_size as usize {
                // 段落大小合适
                let chunk = DocumentChunk {
                    id: Uuid::new_v4(),
                    document_id: Uuid::new_v4(),
                    content: paragraph.to_string(),
                    chunk_index,
                    start_char: char_position,
                    end_char: char_position + paragraph.len() as u32,
                    metadata: ChunkMetadata {
                        section_title: None,
                        page_number: None,
                        paragraph_index: Some(para_index as u32),
                        semantic_type: Some("paragraph".to_string()),
                        confidence_score: 0.95,
                    },
                };
                chunks.push(chunk);
                chunk_index += 1;
            } else {
                // 段落太大，按句子分割
                let sentence_chunks = self.chunk_by_sentence(paragraph).await?;
                for mut chunk in sentence_chunks {
                    chunk.chunk_index = chunk_index;
                    chunk.start_char += char_position;
                    chunk.end_char += char_position;
                    chunk.metadata.paragraph_index = Some(para_index as u32);
                    chunk.metadata.semantic_type = Some("hybrid".to_string());
                    chunks.push(chunk);
                    chunk_index += 1;
                }
            }

            char_position += paragraph.len() as u32 + 2;
        }

        Ok(chunks)
    }
}

/// 语言检测 (简化实现)
fn detect_language(text: &str) -> Option<String> {
    // 简化的语言检测 - 实际项目应该使用专门的语言检测库
    let sample = text.chars().take(1000).collect::<String>();

    // 检测中文
    if sample.chars().any(|c| '\u{4e00}' <= c && c <= '\u{9fff}') {
        return Some("zh".to_string());
    }

    // 检测日文
    if sample.chars().any(|c| {
        ('\u{3040}' <= c && c <= '\u{309f}') || // Hiragana
            ('\u{30a0}' <= c && c <= '\u{30ff}')    // Katakana
    }) {
        return Some("ja".to_string());
    }

    // 检测韩文
    if sample.chars().any(|c| '\u{ac00}' <= c && c <= '\u{d7af}') {
        return Some("ko".to_string());
    }

    // 检测阿拉伯文
    if sample.chars().any(|c| '\u{0600}' <= c && c <= '\u{06ff}') {
        return Some("ar".to_string());
    }

    // 检测俄文
    if sample.chars().any(|c| '\u{0400}' <= c && c <= '\u{04ff}') {
        return Some("ru".to_string());
    }

    // 默认英文
    Some("en".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_text_processor() {
        let processor = TextProcessor::new();
        let data = b"Hello, world!";

        let text = processor.extract_text(data, &DocumentFormat::PlainText).await.unwrap();
        assert_eq!(text, "Hello, world!");

        let metadata = processor.extract_metadata(data, &DocumentFormat::PlainText).await.unwrap();
        assert_eq!(metadata.word_count, Some(2));
        assert_eq!(metadata.char_count, 13);
    }

    #[tokio::test]
    async fn test_document_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedSize,
            max_chunk_size: 10,
            chunk_overlap: 2,
            min_chunk_size: 5,
            respect_sentence_boundaries: false,
            respect_paragraph_boundaries: false,
            preserve_structure: false,
        };

        let chunker = DocumentChunker::new(config);
        let text = "This is a test document with some content to chunk.";

        let chunks = chunker.chunk_text(text, &DocumentFormat::PlainText).await.unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].content.len() <= 10);
    }

    #[tokio::test]
    async fn test_format_detection() {
        let processor = MainDocumentProcessor::new(ChunkingConfig::default());

        // Test PDF detection
        let pdf_data = b"%PDF-1.4";
        let format = processor.detect_format(pdf_data, None).unwrap();
        assert_eq!(format, DocumentFormat::Pdf);

        // Test JSON detection
        let json_data = b"{ \"key\": \"value\" }";
        let format = processor.detect_format(json_data, None).unwrap();
        assert_eq!(format, DocumentFormat::Json);

        // Test filename-based detection
        let format = processor.detect_format(b"content", Some("test.md")).unwrap();
        assert_eq!(format, DocumentFormat::Markdown);
    }

    #[tokio::test]
    async fn test_markdown_processor() {
        let processor = MarkdownProcessor::new();
        let markdown_data = b"# Title\n\nThis is **bold** text with *italic* formatting.\n\n```rust\ncode block\n```";

        let text = processor.extract_text(markdown_data, &DocumentFormat::Markdown).await.unwrap();
        assert!(!text.contains("**"));
        assert!(!text.contains("```"));

        let metadata = processor.extract_metadata(markdown_data, &DocumentFormat::Markdown).await.unwrap();
        assert_eq!(metadata.title, Some("Title".to_string()));
    }

    #[tokio::test]
    async fn test_language_detection() {
        assert_eq!(detect_language("Hello world"), Some("en".to_string()));
        assert_eq!(detect_language("你好世界"), Some("zh".to_string()));
        assert_eq!(detect_language("こんにちは"), Some("ja".to_string()));
        assert_eq!(detect_language("안녕하세요"), Some("ko".to_string()));
    }

    #[tokio::test]
    async fn test_sentence_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Sentence,
            max_chunk_size: 50,
            chunk_overlap: 0,
            min_chunk_size: 10,
            respect_sentence_boundaries: true,
            respect_paragraph_boundaries: false,
            preserve_structure: false,
        };

        let chunker = DocumentChunker::new(config);
        let text = "First sentence. Second sentence! Third sentence? Fourth sentence.";

        let chunks = chunker.chunk_text(text, &DocumentFormat::PlainText).await.unwrap();
        assert!(!chunks.is_empty());

        // Should have multiple chunks due to sentence boundaries
        for chunk in &chunks {
            assert!(chunk.content.len() <= 50);
            assert_eq!(chunk.metadata.semantic_type, Some("sentence".to_string()));
        }
    }

    #[tokio::test]
    async fn test_paragraph_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Paragraph,
            max_chunk_size: 100,
            chunk_overlap: 0,
            min_chunk_size: 10,
            respect_sentence_boundaries: false,
            respect_paragraph_boundaries: true,
            preserve_structure: true,
        };

        let chunker = DocumentChunker::new(config);
        let text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here.";

        let chunks = chunker.chunk_text(text, &DocumentFormat::PlainText).await.unwrap();
        assert_eq!(chunks.len(), 3);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.paragraph_index, Some(i as u32));
            assert_eq!(chunk.metadata.semantic_type, Some("paragraph".to_string()));
        }
    }

    #[tokio::test]
    async fn test_hybrid_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Hybrid,
            max_chunk_size: 30,
            chunk_overlap: 5,
            min_chunk_size: 10,
            respect_sentence_boundaries: true,
            respect_paragraph_boundaries: true,
            preserve_structure: true,
        };

        let chunker = DocumentChunker::new(config);
        let text = "Short paragraph.\n\nThis is a very long paragraph that exceeds the maximum chunk size and should be split into multiple chunks using sentence boundaries. It contains multiple sentences that should be handled appropriately.";

        let chunks = chunker.chunk_text(text, &DocumentFormat::PlainText).await.unwrap();
        assert!(chunks.len() > 2);

        // First chunk should be the short paragraph
        assert!(chunks[0].content.contains("Short paragraph"));
        assert_eq!(chunks[0].metadata.semantic_type, Some("paragraph".to_string()));

        // Remaining chunks should be from the long paragraph
        for chunk in &chunks[1..] {
            assert!(chunk.content.len() <= 30);
            assert_eq!(chunk.metadata.semantic_type, Some("hybrid".to_string()));
        }
    }

    #[tokio::test]
    async fn test_complete_document_processing() {
        let config = ChunkingConfig::default();
        let processor = MainDocumentProcessor::new(config);

        let content = "# Test Document\n\nThis is a test document with multiple paragraphs.\n\nEach paragraph should be processed correctly.";
        let data = content.as_bytes().to_vec();
        let filename = "test.md".to_string();

        let processed = processor.process_document(data, filename, None).await.unwrap();

        assert_eq!(processed.format, DocumentFormat::Markdown);
        assert!(!processed.content.is_empty());
        assert!(!processed.chunks.is_empty());
        assert_eq!(processed.metadata.title, Some("Test Document".to_string()));
        assert!(processed.processing_stats.chunks_created > 0);
        assert!(processed.processing_stats.processing_time_ms > 0);
    }
}