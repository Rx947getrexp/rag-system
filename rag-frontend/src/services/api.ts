// rag-frontend/src/services/api.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

// API 响应接口
export interface ApiResponse<T = any> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
        details?: Record<string, any>;
    };
    message?: string;
    timestamp: string;
    request_id?: string;
}

// 分页响应接口
export interface PaginatedResponse<T = any> {
    success: boolean;
    data: T[];
    pagination: {
        page: number;
        page_size: number;
        total: number;
        total_pages: number;
        has_next: boolean;
        has_prev: boolean;
    };
    timestamp: string;
}

// 文档接口
export interface Document {
    id: string;
    filename: string;
    format: string;
    file_size: number;
    chunk_count: number;
    created_at: string;
    updated_at: string;
    metadata: Record<string, string>;
}

// 搜索结果接口
export interface SearchResultItem {
    id: string;
    content: string;
    score: number;
    metadata: Record<string, string>;
    document_id?: string;
    chunk_index?: number;
}

// 聊天消息接口
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'function';
    content: string;
    metadata?: Record<string, string>;
}

// 聊天请求接口
export interface ChatRequest {
    question: string;
    conversation_history?: ChatMessage[];
    search_options?: {
        strategy?: string;
        top_k?: number;
        similarity_threshold?: number;
        enable_reranking?: boolean;
        rerank_top_k?: number;
        workspace_id?: string;
        filters?: Array<{
            field: string;
            operator: string;
            value: string;
        }>;
    };
    stream?: boolean;
}

// API 客户端类
export class RagApiClient {
    private client: AxiosInstance;
    private baseURL: string;

    constructor(baseURL: string = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // 请求拦截器
        this.client.interceptors.request.use(
            (config) => {
                // 添加请求 ID
                config.headers['X-Request-ID'] = this.generateRequestId();

                // 添加用户 token (如果有)
                const token = localStorage.getItem('auth_token');
                if (token) {
                    config.headers['Authorization'] = `Bearer ${token}`;
                }

                console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data);
                return config;
            },
            (error) => {
                console.error('[API] Request error:', error);
                return Promise.reject(error);
            }
        );

        // 响应拦截器
        this.client.interceptors.response.use(
            (response) => {
                console.log(`[API] Response:`, response.data);
                return response;
            },
            (error) => {
                console.error('[API] Response error:', error.response?.data || error.message);

                // 统一错误处理
                if (error.response?.status === 401) {
                    // 未授权，清除 token 并重定向到登录页
                    localStorage.removeItem('auth_token');
                    window.location.href = '/login';
                }

                return Promise.reject(error);
            }
        );
    }

    // 生成请求 ID
    private generateRequestId(): string {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // 通用请求方法
    private async request<T>(config: AxiosRequestConfig): Promise<T> {
        try {
            const response: AxiosResponse<T> = await this.client.request(config);
            return response.data;
        } catch (error: any) {
            throw this.handleError(error);
        }
    }

    // 错误处理
    private handleError(error: any): Error {
        if (error.response) {
            const { status, data } = error.response;
            const errorMsg = data?.error?.message || data?.message || `HTTP ${status} Error`;
            return new Error(errorMsg);
        } else if (error.request) {
            return new Error('网络连接失败，请检查网络设置');
        } else {
            return new Error(error.message || '未知错误');
        }
    }

    // === 文档管理 API ===

    // 上传文档
    async uploadDocument(
        file: File,
        metadata?: {
            title?: string;
            description?: string;
            tags?: string;
            workspace_id?: string;
            [key: string]: string | undefined;
        }
    ): Promise<ApiResponse> {
        const formData = new FormData();
        formData.append('file', file);

        if (metadata) {
            Object.entries(metadata).forEach(([key, value]) => {
                if (value !== undefined) {
                    formData.append(key, value);
                }
            });
        }

        return this.request<ApiResponse>({
            method: 'POST',
            url: '/api/v1/documents',
            data: formData,
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
    }

    // 获取文档列表
    async listDocuments(
        page: number = 1,
        pageSize: number = 10,
        workspaceId?: string
    ): Promise<PaginatedResponse<Document>> {
        const params: Record<string, any> = { page, page_size: pageSize };
        if (workspaceId) params.workspace_id = workspaceId;

        return this.request<PaginatedResponse<Document>>({
            method: 'GET',
            url: '/api/v1/documents',
            params,
        });
    }

    // 获取文档详情
    async getDocument(documentId: string): Promise<ApiResponse<Document>> {
        return this.request<ApiResponse<Document>>({
            method: 'GET',
            url: `/api/v1/documents/${documentId}`,
        });
    }

    // 删除文档
    async deleteDocument(documentId: string): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'DELETE',
            url: `/api/v1/documents/${documentId}`,
        });
    }

    // 重新索引文档
    async reindexDocument(documentId: string): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'POST',
            url: `/api/v1/documents/${documentId}/reindex`,
        });
    }

    // === 搜索 API ===

    // 执行搜索
    async search(params: {
        query: string;
        strategy?: string;
        top_k?: number;
        similarity_threshold?: number;
        enable_reranking?: boolean;
        rerank_top_k?: number;
        workspace_id?: string;
        filters?: Array<{
            field: string;
            operator: string;
            value: string;
        }>;
    }): Promise<ApiResponse<{
        query: string;
        results: SearchResultItem[];
        total_found: number;
        processing_time_ms: number;
        strategy_used: string;
        metadata: Record<string, string>;
    }>> {
        return this.request({
            method: 'POST',
            url: '/api/v1/search',
            data: params,
        });
    }

    // 搜索建议
    async searchSuggestions(partialQuery: string, limit: number = 5): Promise<ApiResponse<{
        suggestions: string[];
        count: number;
    }>> {
        return this.request({
            method: 'POST',
            url: '/api/v1/search/suggest',
            params: { q: partialQuery, limit },
        });
    }

    // 查找相似内容
    async findSimilar(documentId: string, topK: number = 5): Promise<ApiResponse<{
        document_id: string;
        results: SearchResultItem[];
        count: number;
    }>> {
        return this.request({
            method: 'POST',
            url: '/api/v1/search/similar',
            params: { document_id: documentId, top_k: topK },
        });
    }

    // === 聊天 API ===

    // 聊天补全
    async chatCompletion(request: ChatRequest): Promise<ApiResponse<{
        answer: string;
        conversation_id: string;
        sources: SearchResultItem[];
        usage: {
            prompt_tokens: number;
            completion_tokens: number;
            total_tokens: number;
        };
        processing_time_ms: number;
    }>> {
        return this.request({
            method: 'POST',
            url: '/api/v1/chat',
            data: request,
        });
    }

    // 流式聊天 (返回 EventSource)
    createChatStream(request: ChatRequest): EventSource {
        const url = new URL('/api/v1/chat/stream', this.baseURL);

        // 注意：这里简化了实现，实际应该通过 POST 请求体发送数据
        // 但 EventSource 只支持 GET，所以这里需要特殊处理
        const eventSource = new EventSource(url.toString());

        // 发送聊天请求 (需要通过其他方式实现)
        fetch(url.toString(), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        return eventSource;
    }

    // === WebSocket 连接 ===

    // 创建 WebSocket 连接
    createWebSocket(): WebSocket {
        const wsUrl = this.baseURL.replace('http', 'ws') + '/ws';
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('[WebSocket] 连接已建立');
        };

        ws.onclose = (event) => {
            console.log('[WebSocket] 连接已关闭', event.code, event.reason);
        };

        ws.onerror = (error) => {
            console.error('[WebSocket] 连接错误:', error);
        };

        return ws;
    }

    // === 系统管理 API ===

    // 健康检查
    async healthCheck(): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'GET',
            url: '/health',
        });
    }

    // 就绪检查
    async readinessCheck(): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'GET',
            url: '/ready',
        });
    }

    // 获取系统统计
    async getSystemStats(): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'GET',
            url: '/api/v1/admin/stats',
        });
    }

    // 获取系统健康状态
    async getSystemHealth(): Promise<ApiResponse> {
        return this.request<ApiResponse>({
            method: 'GET',
            url: '/api/v1/admin/health',
        });
    }
}

// 默认导出 API 客户端实例
export const apiClient = new RagApiClient();

// 工具函数
export const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    });
};

export const getFileTypeIcon = (format: string): string => {
    const iconMap: Record<string, string> = {
        'PlainText': '📄',
        'Markdown': '📝',
        'Html': '🌐',
        'Pdf': '📕',
        'Docx': '📘',
        'Json': '📋',
        'Csv': '📊',
    };

    return iconMap[format] || '📄';
};