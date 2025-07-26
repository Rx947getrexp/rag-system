// rag-api/internal/handlers/documents.go
package handlers

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"rag-api/internal/clients"
	"rag-api/pkg/logger"
	pb "rag-api/pkg/pb"

	"github.com/gin-gonic/gin"
)

// DocumentHandlers 文档处理器
type DocumentHandlers struct {
	ragClient *clients.RagEngineClient
}

// NewDocumentHandlers 创建文档处理器
func NewDocumentHandlers(ragClient *clients.RagEngineClient) *DocumentHandlers {
	return &DocumentHandlers{
		ragClient: ragClient,
	}
}

// UploadDocumentRequest 文档上传请求
type UploadDocumentRequest struct {
	Title       string            `form:"title" json:"title"`
	Description string            `form:"description" json:"description"`
	Tags        string            `form:"tags" json:"tags"`
	WorkspaceID string            `form:"workspace_id" json:"workspace_id"`
	Metadata    map[string]string `form:"metadata" json:"metadata"`
}

// UploadDocument 上传文档
// @Summary 上传文档
// @Description 上传并处理文档，支持多种格式
// @Tags 文档管理
// @Accept multipart/form-data
// @Produce json
// @Param file formData file true "文档文件"
// @Param title formData string false "文档标题"
// @Param description formData string false "文档描述"
// @Param tags formData string false "标签，用逗号分隔"
// @Param workspace_id formData string false "工作空间ID"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/documents [post]
func (h *DocumentHandlers) UploadDocument(c *gin.Context) {
	logger.Info("文档上传请求", "client_ip", c.ClientIP())

	// 解析表单数据
	var req UploadDocumentRequest
	if err := c.ShouldBind(&req); err != nil {
		logger.Error("解析上传请求失败", "error", err)
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "INVALID_REQUEST",
				Message: "请求参数无效: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 获取上传的文件
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		logger.Error("获取上传文件失败", "error", err)
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "FILE_REQUIRED",
				Message: "请选择要上传的文件",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}
	defer file.Close()

	// 验证文件大小 (限制为 50MB)
	const maxFileSize = 50 * 1024 * 1024
	if header.Size > maxFileSize {
		logger.Error("文件过大", "size", header.Size, "max_size", maxFileSize)
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "FILE_TOO_LARGE",
				Message: fmt.Sprintf("文件大小不能超过 %d MB", maxFileSize/(1024*1024)),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 验证文件类型
	allowedTypes := []string{
		"text/plain", "text/markdown", "text/csv",
		"application/pdf", "application/json",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/msword", "text/html",
	}

	contentType := header.Header.Get("Content-Type")
	if contentType == "" {
		// 根据文件扩展名推断类型
		ext := strings.ToLower(header.Filename[strings.LastIndex(header.Filename, ".")+1:])
		switch ext {
		case "txt":
			contentType = "text/plain"
		case "md", "markdown":
			contentType = "text/markdown"
		case "pdf":
			contentType = "application/pdf"
		case "docx":
			contentType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
		case "doc":
			contentType = "application/msword"
		case "html", "htm":
			contentType = "text/html"
		case "json":
			contentType = "application/json"
		case "csv":
			contentType = "text/csv"
		default:
			contentType = "application/octet-stream"
		}
	}

	isAllowed := false
	for _, allowedType := range allowedTypes {
		if contentType == allowedType {
			isAllowed = true
			break
		}
	}

	if !isAllowed {
		logger.Error("不支持的文件类型", "content_type", contentType, "filename", header.Filename)
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "UNSUPPORTED_FILE_TYPE",
				Message: fmt.Sprintf("不支持的文件类型: %s", contentType),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 读取文件内容
	data, err := io.ReadAll(file)
	if err != nil {
		logger.Error("读取文件内容失败", "error", err, "filename", header.Filename)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "FILE_READ_ERROR",
				Message: "读取文件失败",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 准备元数据
	metadata := make(map[string]string)
	if req.Title != "" {
		metadata["title"] = req.Title
	} else {
		metadata["title"] = header.Filename
	}
	if req.Description != "" {
		metadata["description"] = req.Description
	}
	if req.Tags != "" {
		metadata["tags"] = req.Tags
	}
	if req.WorkspaceID != "" {
		metadata["workspace_id"] = req.WorkspaceID
	}

	metadata["original_filename"] = header.Filename
	metadata["content_type"] = contentType
	metadata["file_size"] = fmt.Sprintf("%d", header.Size)
	metadata["uploaded_by"] = c.GetHeader("X-User-ID") // 如果有用户认证
	metadata["uploaded_at"] = time.Now().Format(time.RFC3339)

	// 合并自定义元数据
	for k, v := range req.Metadata {
		metadata[k] = v
	}

	// 调用 RAG 引擎处理文档
	logger.Info("调用 RAG 引擎处理文档",
		"filename", header.Filename,
		"size", len(data),
		"content_type", contentType,
	)

	resp, err := h.ragClient.ProcessDocument(c.Request.Context(), data, header.Filename, metadata)
	if err != nil {
		logger.Error("RAG 引擎处理文档失败", "error", err, "filename", header.Filename)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "PROCESSING_ERROR",
				Message: "文档处理失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 检查处理结果
	if !resp.Success {
		logger.Error("文档处理失败",
			"document_id", resp.DocumentId,
			"errors", resp.Errors,
			"warnings", resp.Warnings,
		)

		errorMsg := "文档处理失败"
		if len(resp.Errors) > 0 {
			errorMsg = strings.Join(resp.Errors, "; ")
		}

		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "PROCESSING_FAILED",
				Message: errorMsg,
				Details: map[string]interface{}{
					"document_id": resp.DocumentId,
					"errors":      resp.Errors,
					"warnings":    resp.Warnings,
				},
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 处理成功
	logger.Info("文档处理成功",
		"document_id", resp.DocumentId,
		"chunks", resp.ChunksProcessed,
		"vectors", resp.VectorsCreated,
		"processing_time_ms", resp.ProcessingTimeMs,
	)

	c.JSON(http.StatusOK, Response{
		Success: true,
		Data: map[string]interface{}{
			"document_id":        resp.DocumentId,
			"filename":           header.Filename,
			"chunks_processed":   resp.ChunksProcessed,
			"vectors_created":    resp.VectorsCreated,
			"processing_time_ms": resp.ProcessingTimeMs,
			"warnings":           resp.Warnings,
			"metadata":           resp.Metadata,
		},
		Message:   "文档上传和处理成功",
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}

// GetDocument 获取文档详情
// @Summary 获取文档详情
// @Description 根据文档ID获取文档详细信息
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param id path string true "文档ID"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Router /api/v1/documents/{id} [get]
func (h *DocumentHandlers) GetDocument(c *gin.Context) {
	documentID := c.Param("id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Debug("获取文档详情", "document_id", documentID)

	resp, err := h.ragClient.GetDocument(c.Request.Context(), documentID)
	if err != nil {
		logger.Error("获取文档详情失败", "document_id", documentID, "error", err)
		c.JSON(http.StatusNotFound, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "DOCUMENT_NOT_FOUND",
				Message: "文档不存在或获取失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Success:   true,
		Data:      resp.Document,
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}

// ListDocuments 列出文档
// @Summary 列出文档
// @Description 分页获取文档列表
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param page query int false "页码" default(1)
// ListDocuments 列出文档
// @Summary 列出文档
// @Description 分页获取文档列表
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param page query int false "页码" default(1)
// @Param page_size query int false "每页数量" default(10)
// @Param workspace_id query string false "工作空间ID"
// @Success 200 {object} PaginatedResponse
// @Failure 400 {object} Response
// @Router /api/v1/documents [get]
func (h *DocumentHandlers) ListDocuments(c *gin.Context) {
	// 解析查询参数
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "10")
	workspaceID := c.Query("workspace_id")

	page, err := strconv.ParseUint(pageStr, 10, 32)
	if err != nil || page < 1 {
		page = 1
	}

	pageSize, err := strconv.ParseUint(pageSizeStr, 10, 32)
	if err != nil || pageSize < 1 || pageSize > 100 {
		pageSize = 10
	}

	logger.Debug("列出文档", "page", page, "page_size", pageSize, "workspace_id", workspaceID)

	resp, err := h.ragClient.ListDocuments(c.Request.Context(), uint32(page), uint32(pageSize), workspaceID)
	if err != nil {
		logger.Error("列出文档失败", "error", err)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "LIST_DOCUMENTS_ERROR",
				Message: "获取文档列表失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	totalPages := int((resp.Total + uint64(pageSize) - 1) / uint64(pageSize))

	c.JSON(http.StatusOK, PaginatedResponse{
		Success: true,
		Data:    resp.Documents,
		Pagination: Pagination{
			Page:       int(page),
			PageSize:   int(pageSize),
			Total:      int64(resp.Total),
			TotalPages: totalPages,
			HasNext:    int(page) < totalPages,
			HasPrev:    int(page) > 1,
		},
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

// UpdateDocument 更新文档
// @Summary 更新文档
// @Description 更新文档元数据
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param id path string true "文档ID"
// @Param request body map[string]string true "更新数据"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Router /api/v1/documents/{id} [put]
func (h *DocumentHandlers) UpdateDocument(c *gin.Context) {
	documentID := c.Param("id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	var updateData map[string]string
	if err := c.ShouldBindJSON(&updateData); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "INVALID_REQUEST",
				Message: "请求数据格式错误: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Info("更新文档", "document_id", documentID, "updates", updateData)

	// 简化实现 - 实际应该调用 RAG 引擎的更新接口
	c.JSON(http.StatusNotImplemented, Response{
		Success: false,
		Error: &ErrorInfo{
			Code:    "NOT_IMPLEMENTED",
			Message: "文档更新功能尚未实现",
		},
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

// DeleteDocument 删除文档
// @Summary 删除文档
// @Description 删除指定的文档及其所有相关数据
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param id path string true "文档ID"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Router /api/v1/documents/{id} [delete]
func (h *DocumentHandlers) DeleteDocument(c *gin.Context) {
	documentID := c.Param("id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Info("删除文档", "document_id", documentID)

	resp, err := h.ragClient.DeleteDocument(c.Request.Context(), documentID)
	if err != nil {
		logger.Error("删除文档失败", "document_id", documentID, "error", err)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "DELETE_ERROR",
				Message: "删除文档失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	if !resp.Success {
		c.JSON(http.StatusNotFound, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "DELETE_FAILED",
				Message: resp.Message,
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Info("文档删除成功", "document_id", documentID)

	c.JSON(http.StatusOK, Response{
		Success: true,
		Data: map[string]interface{}{
			"document_id": documentID,
			"message":     resp.Message,
		},
		Message:   "文档删除成功",
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}

// GetDocumentChunks 获取文档块
// @Summary 获取文档块
// @Description 获取文档的所有文本块
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param id path string true "文档ID"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Router /api/v1/documents/{id}/chunks [get]
func (h *DocumentHandlers) GetDocumentChunks(c *gin.Context) {
	documentID := c.Param("id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Debug("获取文档块", "document_id", documentID)

	// 简化实现 - 实际应该调用 RAG 引擎获取文档块
	c.JSON(http.StatusNotImplemented, Response{
		Success: false,
		Error: &ErrorInfo{
			Code:    "NOT_IMPLEMENTED",
			Message: "获取文档块功能尚未实现",
		},
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

// ReindexDocument 重新索引文档
// @Summary 重新索引文档
// @Description 重新处理和索引指定文档
// @Tags 文档管理
// @Accept json
// @Produce json
// @Param id path string true "文档ID"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Router /api/v1/documents/{id}/reindex [post]
func (h *DocumentHandlers) ReindexDocument(c *gin.Context) {
	documentID := c.Param("id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Info("重新索引文档", "document_id", documentID)

	// 简化实现 - 实际应该调用 RAG 引擎重新索引
	c.JSON(http.StatusNotImplemented, Response{
		Success: false,
		Error: &ErrorInfo{
			Code:    "NOT_IMPLEMENTED",
			Message: "重新索引功能尚未实现",
		},
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

// SearchHandlers 搜索处理器
type SearchHandlers struct {
	ragClient *clients.RagEngineClient
}

// NewSearchHandlers 创建搜索处理器
func NewSearchHandlers(ragClient *clients.RagEngineClient) *SearchHandlers {
	return &SearchHandlers{
		ragClient: ragClient,
	}
}

// SearchRequest 搜索请求
type SearchRequest struct {
	Query               string           `json:"query" binding:"required"`
	Strategy            string           `json:"strategy,omitempty"`
	TopK                uint32           `json:"top_k,omitempty"`
	SimilarityThreshold *float32         `json:"similarity_threshold,omitempty"`
	EnableReranking     bool             `json:"enable_reranking,omitempty"`
	RerankTopK          *uint32          `json:"rerank_top_k,omitempty"`
	WorkspaceID         string           `json:"workspace_id,omitempty"`
	Filters             []QueryFilterReq `json:"filters,omitempty"`
}

// QueryFilterReq 查询过滤器请求
type QueryFilterReq struct {
	Field    string `json:"field" binding:"required"`
	Operator string `json:"operator" binding:"required"`
	Value    string `json:"value" binding:"required"`
}

// Search 执行搜索
// @Summary 执行搜索
// @Description 在文档库中搜索相关内容
// @Tags 搜索
// @Accept json
// @Produce json
// @Param request body SearchRequest true "搜索请求"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/search [post]
func (h *SearchHandlers) Search(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Error("解析搜索请求失败", "error", err)
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "INVALID_REQUEST",
				Message: "请求参数无效: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	// 设置默认值
	if req.Strategy == "" {
		req.Strategy = "hybrid"
	}
	if req.TopK == 0 {
		req.TopK = 10
	}

	logger.Info("执行搜索", "query", req.Query, "strategy", req.Strategy, "top_k", req.TopK)

	// 构建搜索选项
	options := &pb.SearchOptions{
		Strategy:            req.Strategy,
		TopK:                req.TopK,
		SimilarityThreshold: req.SimilarityThreshold,
		EnableReranking:     req.EnableReranking,
		RerankTopK:          req.RerankTopK,
		WorkspaceId:         req.WorkspaceID,
	}

	// 添加过滤器
	for _, filter := range req.Filters {
		options.Filters = append(options.Filters, &pb.QueryFilter{
			Field:    filter.Field,
			Operator: filter.Operator,
			Value:    filter.Value,
		})
	}

	// 调用 RAG 引擎搜索
	resp, err := h.ragClient.Search(c.Request.Context(), req.Query, options)
	if err != nil {
		logger.Error("搜索失败", "query", req.Query, "error", err)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "SEARCH_ERROR",
				Message: "搜索失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	logger.Info("搜索完成",
		"query", req.Query,
		"results", len(resp.Results),
		"total_found", resp.TotalFound,
		"processing_time_ms", resp.ProcessingTimeMs,
	)

	c.JSON(http.StatusOK, Response{
		Success: true,
		Data: map[string]interface{}{
			"query":              req.Query,
			"results":            resp.Results,
			"total_found":        resp.TotalFound,
			"processing_time_ms": resp.ProcessingTimeMs,
			"strategy_used":      resp.StrategyUsed,
			"metadata":           resp.Metadata,
		},
		Message:   fmt.Sprintf("找到 %d 个相关结果", len(resp.Results)),
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}

// SearchSuggestions 搜索建议
// @Summary 搜索建议
// @Description 根据部分查询生成搜索建议
// @Tags 搜索
// @Accept json
// @Produce json
// @Param q query string true "部分查询"
// @Param limit query int false "建议数量限制" default(5)
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Router /api/v1/search/suggest [post]
func (h *SearchHandlers) SearchSuggestions(c *gin.Context) {
	partialQuery := c.Query("q")
	if partialQuery == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_QUERY",
				Message: "查询参数不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	limitStr := c.DefaultQuery("limit", "5")
	limit, err := strconv.ParseUint(limitStr, 10, 32)
	if err != nil || limit > 20 {
		limit = 5
	}

	logger.Debug("生成搜索建议", "partial_query", partialQuery, "limit", limit)

	resp, err := h.ragClient.SuggestQueries(c.Request.Context(), partialQuery, uint32(limit))
	if err != nil {
		logger.Error("生成搜索建议失败", "partial_query", partialQuery, "error", err)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "SUGGESTION_ERROR",
				Message: "生成搜索建议失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Success: true,
		Data: map[string]interface{}{
			"suggestions": resp.Suggestions,
			"count":       len(resp.Suggestions),
		},
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}

// FindSimilar 查找相似内容
// @Summary 查找相似内容
// @Description 查找与指定文档相似的内容
// @Tags 搜索
// @Accept json
// @Produce json
// @Param document_id query string true "文档ID"
// @Param top_k query int false "返回数量" default(5)
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Router /api/v1/search/similar [post]
func (h *SearchHandlers) FindSimilar(c *gin.Context) {
	documentID := c.Query("document_id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "MISSING_DOCUMENT_ID",
				Message: "文档ID不能为空",
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	topKStr := c.DefaultQuery("top_k", "5")
	topK, err := strconv.ParseUint(topKStr, 10, 32)
	if err != nil || topK > 50 {
		topK = 5
	}

	logger.Debug("查找相似内容", "document_id", documentID, "top_k", topK)

	resp, err := h.ragClient.FindSimilar(c.Request.Context(), documentID, uint32(topK))
	if err != nil {
		logger.Error("查找相似内容失败", "document_id", documentID, "error", err)
		c.JSON(http.StatusInternalServerError, Response{
			Success: false,
			Error: &ErrorInfo{
				Code:    "SIMILAR_SEARCH_ERROR",
				Message: "查找相似内容失败: " + err.Error(),
			},
			Timestamp: time.Now().Format(time.RFC3339),
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Success: true,
		Data: map[string]interface{}{
			"document_id": documentID,
			"results":     resp.Results,
			"count":       len(resp.Results),
		},
		Message:   fmt.Sprintf("找到 %d 个相似内容", len(resp.Results)),
		Timestamp: time.Now().Format(time.RFC3339),
		RequestID: c.GetHeader("X-Request-ID"),
	})
}
