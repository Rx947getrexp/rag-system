// rag-frontend/src/App.tsx
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, Menu, Typography, Spin, message, Button, Drawer } from 'antd';
import {
    UploadOutlined,
    SearchOutlined,
    MessageOutlined,
    SettingOutlined,
    DashboardOutlined,
    MenuOutlined,
    CloseOutlined,
} from '@ant-design/icons';

// 组件导入
import DocumentUpload from './components/DocumentUpload';
import SearchInterface from './components/SearchInterface';
import ChatInterface from './components/ChatInterface';
import Dashboard from './components/Dashboard';
import SystemHealth from './components/SystemHealth';

// API 客户端
import { apiClient } from './services/api';

// 样式
import './App.css';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

// 主应用组件
const App: React.FC = () => {
    const [selectedKey, setSelectedKey] = useState<string>('dashboard');
    const [loading, setLoading] = useState<boolean>(true);
    const [systemHealth, setSystemHealth] = useState<boolean>(true);
    const [mobileMenuVisible, setMobileMenuVisible] = useState<boolean>(false);
    const [isMobile, setIsMobile] = useState<boolean>(false);

    // 检查屏幕大小
    useEffect(() => {
        const checkScreenSize = () => {
            setIsMobile(window.innerWidth < 768);
        };

        checkScreenSize();
        window.addEventListener('resize', checkScreenSize);

        return () => window.removeEventListener('resize', checkScreenSize);
    }, []);

    // 初始化应用
    useEffect(() => {
        const initializeApp = async () => {
            try {
                setLoading(true);

                // 检查系统健康状态
                const healthResponse = await apiClient.healthCheck();
                if (!healthResponse.success) {
                    setSystemHealth(false);
                    message.error('系统健康检查失败，部分功能可能不可用');
                }

                message.success('系统初始化完成');
            } catch (error: any) {
                console.error('系统初始化失败:', error);
                setSystemHealth(false);
                message.error(`系统初始化失败: ${error.message}`);
            } finally {
                setLoading(false);
            }
        };

        initializeApp();
    }, []);

    // 菜单项配置
    const menuItems = [
        {
            key: 'dashboard',
            icon: <DashboardOutlined />,
            label: '仪表板',
        },
        {
            key: 'upload',
            icon: <UploadOutlined />,
            label: '文档上传',
        },
        {
            key: 'search',
            icon: <SearchOutlined />,
            label: '搜索',
        },
        {
            key: 'chat',
            icon: <MessageOutlined />,
            label: '智能问答',
        },
        {
            key: 'health',
            icon: <SettingOutlined />,
            label: '系统状态',
        },
    ];

    // 渲染内容
    const renderContent = () => {
        switch (selectedKey) {
            case 'dashboard':
                return <Dashboard />;
            case 'upload':
                return <DocumentUpload />;
            case 'search':
                return <SearchInterface />;
            case 'chat':
                return <ChatInterface />;
            case 'health':
                return <SystemHealth />;
            default:
                return <Dashboard />;
        }
    };

    // 处理菜单点击
    const handleMenuClick = (key: string) => {
        setSelectedKey(key);
        if (isMobile) {
            setMobileMenuVisible(false);
        }
    };

    // 移动端菜单
    const MobileMenu = () => (
        <Drawer
            title="菜单"
            placement="left"
            onClose={() => setMobileMenuVisible(false)}
            open={mobileMenuVisible}
            bodyStyle={{ padding: 0 }}
        >
            <Menu
                mode="inline"
                selectedKeys={[selectedKey]}
                style={{ border: 'none' }}
                items={menuItems.map(item => ({
                    ...item,
                    onClick: () => handleMenuClick(item.key),
                }))}
            />
        </Drawer>
    );

    if (loading) {
        return (
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100vh',
                flexDirection: 'column'
            }}>
                <Spin size="large" />
                <p style={{ marginTop: 16, color: '#666' }}>正在初始化 RAG 系统...</p>
            </div>
        );
    }

    return (
        <Router>
            <Layout style={{ minHeight: '100vh' }}>
                {/* 移动端菜单 */}
                {isMobile && <MobileMenu />}

                {/* 桌面端侧边栏 */}
                {!isMobile && (
                    <Sider
                        width={250}
                        style={{
                            background: '#fff',
                            boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
                        }}
                    >
                        <div style={{
                            padding: '24px 16px',
                            borderBottom: '1px solid #f0f0f0',
                            textAlign: 'center'
                        }}>
                            <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                                🤖 RAG 系统
                            </Title>
                            <p style={{ margin: '8px 0 0', color: '#666', fontSize: '12px' }}>
                                智能文档问答系统
                            </p>
                        </div>

                        <Menu
                            mode="inline"
                            selectedKeys={[selectedKey]}
                            style={{ border: 'none', marginTop: 16 }}
                            items={menuItems.map(item => ({
                                ...item,
                                onClick: () => handleMenuClick(item.key),
                            }))}
                        />

                        {/* 系统状态指示器 */}
                        <div style={{
                            position: 'absolute',
                            bottom: 16,
                            left: 16,
                            right: 16,
                            padding: 12,
                            background: systemHealth ? '#f6ffed' : '#fff2f0',
                            border: `1px solid ${systemHealth ? '#b7eb8f' : '#ffccc7'}`,
                            borderRadius: 6,
                            fontSize: 12
                        }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8
                            }}>
                                <div style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    background: systemHealth ? '#52c41a' : '#ff4d4f'
                                }} />
                                <span style={{ color: systemHealth ? '#52c41a' : '#ff4d4f' }}>
                  {systemHealth ? '系统正常' : '系统异常'}
                </span>
                            </div>
                        </div>
                    </Sider>
                )}

                <Layout>
                    <Header style={{
                        background: '#fff',
                        padding: '0 24px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                            {/* 移动端菜单按钮 */}
                            {isMobile && (
                                <>
                                    <Button
                                        type="text"
                                        icon={<MenuOutlined />}
                                        onClick={() => setMobileMenuVisible(true)}
                                        style={{ fontSize: 16 }}
                                    />
                                    <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                                        RAG 系统
                                    </Title>
                                </>
                            )}

                            {/* 桌面端标题 */}
                            {!isMobile && (
                                <Title level={4} style={{ margin: 0 }}>
                                    {menuItems.find(item => item.key === selectedKey)?.label}
                                </Title>
                            )}
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                            {/* 系统状态指示器 (移动端) */}
                            {isMobile && (
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 8,
                                    padding: '4px 8px',
                                    background: systemHealth ? '#f6ffed' : '#fff2f0',
                                    border: `1px solid ${systemHealth ? '#b7eb8f' : '#ffccc7'}`,
                                    borderRadius: 4,
                                    fontSize: 12
                                }}>
                                    <div style={{
                                        width: 6,
                                        height: 6,
                                        borderRadius: '50%',
                                        background: systemHealth ? '#52c41a' : '#ff4d4f'
                                    }} />
                                    <span style={{ color: systemHealth ? '#52c41a' : '#ff4d4f' }}>
                    {systemHealth ? '正常' : '异常'}
                  </span>
                                </div>
                            )}
                        </div>
                    </Header>

                    <Content style={{
                        padding: isMobile ? 16 : 24,
                        background: '#f5f5f5',
                        minHeight: 'calc(100vh - 64px)',
                        overflow: 'auto'
                    }}>
                        <Routes>
                            <Route path="/" element={<Navigate to="/dashboard" replace />} />
                            <Route path="/dashboard" element={renderContent()} />
                            <Route path="/upload" element={renderContent()} />
                            <Route path="/search" element={renderContent()} />
                            <Route path="/chat" element={renderContent()} />
                            <Route path="/health" element={renderContent()} />
                            <Route path="*" element={<Navigate to="/dashboard" replace />} />
                        </Routes>
                    </Content>
                </Layout>
            </Layout>
        </Router>
    );
};

export default App;