// rag-frontend/src/components/DocumentUpload.tsx
import React, { useState, useCallback } from 'react';
import {
    Card,
    Upload,
    Button,
    Form,
    Input,
    message,
    Progress,
    Typography,
    Space,
    Tag,
    Divider,
    Alert,
    Row,
    Col,
    List,
    Modal,
} from 'antd';
import {
    UploadOutlined,
    InboxOutlined,
    FileTextOutlined,
    DeleteOutlined,
    EyeOutlined,
    CheckCircleOutlined,
    ExclamationCircleOutlined,
} from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd';

import { apiClient, formatFileSize, getFileTypeIcon } from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;
const { TextArea } = Input;

interface UploadResult {
    success: boolean;
    filename: string;
    document_id?: string;
    chunks_processed?: number;
    vectors_created?: number;
    processing_time_ms?: number;
    errors?: string[];
    warnings?: string[];
}

const DocumentUpload: React.FC = () => {
    const [form] = Form.useForm();
    const [uploading, setUploading] = useState<boolean>(false);
    const [uploadProgress, setUploadProgress] = useState<number>(0);
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
    const [resultModalVisible, setResultModalVisible] = useState<boolean>(false);

    // 支持的文件类型
    const supportedTypes = [
        { ext: '.txt', type: 'text/plain', name: '文本文件' },
        { ext: '.md', type: 'text/markdown', name: 'Markdown' },
        { ext: '.pdf', type: 'application/pdf', name: 'PDF 文档' },
        { ext: '.docx', type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', name: 'Word 文档' },
        { ext: '.doc', type: 'application/msword', name: 'Word 文档 (旧版)' },
        { ext: '.html', type: 'text/html', name: 'HTML 文件' },
        { ext: '.json', type: 'application/json', name: 'JSON 文件' },
        { ext: '.csv', type: 'text/csv', name: 'CSV 文件' },
    ];

    // 文件上传配置
    const uploadProps: UploadProps = {
        multiple: true,
        beforeUpload: (file) => {
            // 检查文件大小 (50MB)
            // 文件上传配置
            const uploadProps: UploadProps = {
                multiple: true,
                beforeUpload: (file) => {
                    // 检查文件大小 (50MB)
                    const maxSize = 50 * 1024 * 1024;
                    if (file.size > maxSize) {
                        message.error(`文件 ${file.name} 超过 50MB 限制`);
                        return false;
                    }

                    // 检查文件类型
                    const isSupported = supportedTypes.some(
                        type => file.type === type.type || file.name.toLowerCase().endsWith(type.ext)
                    );
                    if (!isSupported) {
                        message.error(`不支持的文件类型: ${file.name}`);
                        return false;
                    }

                    return false; // 阻止自动上传
                },
                onRemove: (file) => {
                    setFileList(prev => prev.filter(item => item.uid !== file.uid));
                },
                fileList,
                onChange: ({ fileList: newFileList }) => {
                    setFileList(newFileList);
                },
            };

            // 处理文件上传
            const handleUpload = useCallback(async () => {
                if (fileList.length === 0) {
                    message.warning('请选择要上传的文件');
                    return;
                }

                try {
                    setUploading(true);
                    setUploadProgress(0);
                    const results: UploadResult[] = [];

                    const formValues = await form.validateFields();

                    for (let i = 0; i < fileList.length; i++) {
                        const file = fileList[i];
                        setUploadProgress(Math.round(((i + 0.5) / fileList.length) * 100));

                        try {
                            const metadata = {
                                title: formValues.title || file.name,
                                description: formValues.description,
                                tags: formValues.tags,
                                workspace_id: formValues.workspace_id,
                            };

                            const response = await apiClient.uploadDocument(file.originFileObj as File, metadata);

                            if (response.success) {
                                results.push({
                                    success: true,
                                    filename: file.name,
                                    document_id: response.data?.document_id,
                                    chunks_processed: response.data?.chunks_processed,
                                    vectors_created: response.data?.vectors_created,
                                    processing_time_ms: response.data?.processing_time_ms,
                                    warnings: response.data?.warnings,
                                });
                                message.success(`文件 ${file.name} 上传成功`);
                            } else {
                                results.push({
                                    success: false,
                                    filename: file.name,
                                    errors: [response.error?.message || '上传失败'],
                                });
                                message.error(`文件 ${file.name} 上传失败`);
                            }
                        } catch (error: any) {
                            results.push({
                                success: false,
                                filename: file.name,
                                errors: [error.message || '上传过程中发生错误'],
                            });
                            message.error(`文件 ${file.name} 上传失败: ${error.message}`);
                        }

                        setUploadProgress(Math.round(((i + 1) / fileList.length) * 100));
                    }

                    setUploadResults(results);
                    setResultModalVisible(true);

                    // 清空表单和文件列表
                    if (results.every(r => r.success)) {
                        form.resetFields();
                        setFileList([]);
                    }

                } catch (error: any) {
                    message.error('表单验证失败');
                } finally {
                    setUploading(false);
                    setUploadProgress(0);
                }
            }, [fileList, form]);

            // 清空文件列表
            const clearFiles = () => {
                setFileList([]);
                setUploadResults([]);
            };

            // 渲染上传结果
            const renderUploadResults = () => (
                <Modal
                    title="上传结果"
                    open={resultModalVisible}
                    onCancel={() => setResultModalVisible(false)}
                    footer={[
                        <Button key="close" onClick={() => setResultModalVisible(false)}>
                            关闭
                        </Button>
                    ]}
                    width={700}
                >
                    <List
                        dataSource={uploadResults}
                        renderItem={(result) => (
                            <List.Item>
                                <List.Item.Meta
                                    avatar={
                                        result.success ? (
                                            <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 20 }} />
                                        ) : (
                                            <ExclamationCircleOutlined style={{ color: '#ff4d4f', fontSize: 20 }} />
                                        )
                                    }
                                    title={
                                        <Space>
                                            <Text strong>{result.filename}</Text>
                                            <Tag color={result.success ? 'success' : 'error'}>
                                                {result.success ? '成功' : '失败'}
                                            </Tag>
                                        </Space>
                                    }
                                    description={
                                        <div>
                                            {result.success ? (
                                                <div>
                                                    <Text type="secondary">
                                                        文档ID: {result.document_id}
                                                    </Text>
                                                    <br />
                                                    <Text type="secondary">
                                                        处理块数: {result.chunks_processed},
                                                        向量数: {result.vectors_created},
                                                        耗时: {result.processing_time_ms}ms
                                                    </Text>
                                                    {result.warnings && result.warnings.length > 0 && (
                                                        <div style={{ marginTop: 8 }}>
                                                            <Alert
                                                                message="警告"
                                                                description={result.warnings.join(', ')}
                                                                type="warning"
                                                                showIcon
                                                                size="small"
                                                            />
                                                        </div>
                                                    )}
                                                </div>
                                            ) : (
                                                <div>
                                                    {result.errors?.map((error, index) => (
                                                        <Alert
                                                            key={index}
                                                            message="错误"
                                                            description={error}
                                                            type="error"
                                                            showIcon
                                                            size="small"
                                                            style={{ marginBottom: index < result.errors!.length - 1 ? 8 : 0 }}
                                                        />
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    }
                                />
                            </List.Item>
                        )}
                    />
                </Modal>
            );

            return (
                <div style={{ maxWidth: 1200, margin: '0 auto' }}>
                    <Title level={2}>📤 文档上传</Title>
                    <Paragraph>
                        上传您的文档到 RAG 系统中，支持多种格式的文档，系统会自动进行文本提取、分块和向量化处理。
                    </Paragraph>

                    <Row gutter={24}>
                        <Col xs={24} lg={16}>
                            <Card title="文档上传" style={{ marginBottom: 24 }}>
                                <Form
                                    form={form}
                                    layout="vertical"
                                    initialValues={{
                                        title: '',
                                        description: '',
                                        tags: '',
                                        workspace_id: '',
                                    }}
                                >
                                    <Row gutter={16}>
                                        <Col xs={24} sm={12}>
                                            <Form.Item
                                                name="title"
                                                label="文档标题"
                                                tooltip="如果不填写，将使用文件名作为标题"
                                            >
                                                <Input placeholder="请输入文档标题" />
                                            </Form.Item>
                                        </Col>
                                        <Col xs={24} sm={12}>
                                            <Form.Item
                                                name="tags"
                                                label="标签"
                                                tooltip="用逗号分隔多个标签"
                                            >
                                                <Input placeholder="例如: 技术文档, API, 教程" />
                                            </Form.Item>
                                        </Col>
                                    </Row>

                                    <Form.Item
                                        name="description"
                                        label="文档描述"
                                    >
                                        <TextArea
                                            rows={3}
                                            placeholder="请输入文档描述（可选）"
                                        />
                                    </Form.Item>

                                    <Form.Item
                                        name="workspace_id"
                                        label="工作空间ID"
                                        tooltip="指定文档所属的工作空间"
                                    >
                                        <Input placeholder="可选，留空则使用默认工作空间" />
                                    </Form.Item>
                                </Form>

                                <Divider />

                                <Dragger {...uploadProps} style={{ marginBottom: 16 }}>
                                    <p className="ant-upload-drag-icon">
                                        <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                                    </p>
                                    <p className="ant-upload-text">
                                        点击或拖拽文件到此区域上传
                                    </p>
                                    <p className="ant-upload-hint">
                                        支持单个或批量上传，文件大小限制 50MB
                                    </p>
                                </Dragger>

                                {fileList.length > 0 && (
                                    <div style={{ marginBottom: 16 }}>
                                        <div style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            marginBottom: 8
                                        }}>
                                            <Text strong>已选择文件 ({fileList.length})</Text>
                                            <Button
                                                type="link"
                                                size="small"
                                                icon={<DeleteOutlined />}
                                                onClick={clearFiles}
                                            >
                                                清空
                                            </Button>
                                        </div>

                                        <List
                                            size="small"
                                            bordered
                                            dataSource={fileList}
                                            renderItem={(file) => (
                                                <List.Item
                                                    actions={[
                                                        <Button
                                                            type="link"
                                                            size="small"
                                                            icon={<DeleteOutlined />}
                                                            onClick={() => uploadProps.onRemove?.(file)}
                                                        />
                                                    ]}
                                                >
                                                    <List.Item.Meta
                                                        avatar={<FileTextOutlined />}
                                                        title={file.name}
                                                        description={formatFileSize(file.size || 0)}
                                                    />
                                                </List.Item>
                                            )}
                                        />
                                    </div>
                                )}

                                {uploading && (
                                    <div style={{ marginBottom: 16 }}>
                                        <Text>上传进度:</Text>
                                        <Progress
                                            percent={uploadProgress}
                                            status="active"
                                            style={{ marginTop: 8 }}
                                        />
                                    </div>
                                )}

                                <Space>
                                    <Button
                                        type="primary"
                                        icon={<UploadOutlined />}
                                        onClick={handleUpload}
                                        loading={uploading}
                                        disabled={fileList.length === 0}
                                        size="large"
                                    >
                                        {uploading ? '上传中...' : '开始上传'}
                                    </Button>

                                    <Button
                                        onClick={clearFiles}
                                        disabled={uploading || fileList.length === 0}
                                    >
                                        清空文件
                                    </Button>
                                </Space>
                            </Card>
                        </Col>

                        <Col xs={24} lg={8}>
                            <Card title="📋 支持的文件格式" style={{ marginBottom: 24 }}>
                                <List
                                    size="small"
                                    dataSource={supportedTypes}
                                    renderItem={(type) => (
                                        <List.Item>
                                            <Space>
                                                <Text code>{type.ext}</Text>
                                                <Text>{type.name}</Text>
                                            </Space>
                                        </List.Item>
                                    )}
                                />
                            </Card>

                            <Card title="💡 使用提示">
                                <div style={{ fontSize: 14, lineHeight: 1.6 }}>
                                    <Paragraph>
                                        <Text strong>文档处理流程:</Text>
                                    </Paragraph>
                                    <ol style={{ paddingLeft: 20, margin: 0 }}>
                                        <li>文本提取和清理</li>
                                        <li>智能分块处理</li>
                                        <li>向量化嵌入</li>
                                        <li>存储到向量数据库</li>
                                    </ol>

                                    <Divider style={{ margin: '16px 0' }} />

                                    <Paragraph>
                                        <Text strong>建议:</Text>
                                    </Paragraph>
                                    <ul style={{ paddingLeft: 20, margin: 0 }}>
                                        <li>上传结构化的文档效果更好</li>
                                        <li>添加合适的标题和描述有助于检索</li>
                                        <li>大文件会需要更长的处理时间</li>
                                        <li>使用标签便于后续管理和筛选</li>
                                    </ul>
                                </div>
                            </Card>
                        </Col>
                    </Row>

                    {/* 上传结果弹窗 */}
                    {renderUploadResults()}
                </div>
            );
        };

        export default DocumentUpload;