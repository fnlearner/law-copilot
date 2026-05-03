import React, { useState } from 'react'
import { Layout, Menu, Typography, Badge, Space } from 'antd'
import {
  MessageOutlined,
  SearchOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import ChatPage from './pages/ChatPage'
import SearchPage from './pages/SearchPage'
import DocumentPage from './pages/DocumentPage'

const { Header, Sider, Content } = Layout
const { Title } = Typography

const menuItems = [
  {
    key: 'chat',
    icon: <MessageOutlined />,
    label: '法律问答',
  },
  {
    key: 'search',
    icon: <SearchOutlined />,
    label: '信息检索',
  },
  {
    key: 'document',
    icon: <FileTextOutlined />,
    label: '文档管理',
  },
]

export default function App() {
  const [activeMenu, setActiveMenu] = useState('chat')
  const [collapsed, setCollapsed] = useState(false)

  const renderContent = () => {
    switch (activeMenu) {
      case 'chat':
        return <ChatPage />
      case 'search':
        return <SearchPage />
      case 'document':
        return <DocumentPage />
      default:
        return <ChatPage />
    }
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 左侧导航 */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        width={240}
        style={{
          background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
          borderRight: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        {/* Logo 区域 */}
        <div
          style={{
            height: 72,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: collapsed ? '0' : '0 24px',
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            gap: 12,
          }}
        >
          {/* <ScaleOutlined
            style={{ fontSize: 28, color: '#3b82f6' }}
          spin={false}
        /> */}
          {!collapsed && (
            <Title level={4} style={{ color: '#fff', margin: 0, fontWeight: 600, letterSpacing: '-0.5px' }}>
              LawCopilot
            </Title>
          )}
        </div>

        {/* 菜单 */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[activeMenu]}
          onClick={({ key }) => setActiveMenu(key)}
          items={menuItems}
          style={{
            background: 'transparent',
            borderRight: 'none',
            marginTop: 12,
          }}
        />

        {/* 底部信息 */}
        {!collapsed && (
          <div
            style={{
              position: 'absolute',
              bottom: 20,
              left: 24,
              right: 24,
              padding: '12px 16px',
              borderRadius: 10,
              background: 'rgba(59,130,246,0.1)',
              border: '1px solid rgba(59,130,246,0.2)',
            }}
          >
            <Space direction="vertical" size={4}>
              <span style={{ color: '#94a3b8', fontSize: 11 }}>法律研究 Agent</span>
              <span style={{ color: '#64748b', fontSize: 10 }}>RAG + DeepSeek LLM</span>
            </Space>
          </div>
        )}
      </Sider>

      {/* 右侧内容区 */}
      <Layout>
        <Header
          style={{
            background: '#fff',
            padding: '0 32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid #f0f0f0',
            boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
          }}
        >
          <Typography.Text strong style={{ fontSize: 16, letterSpacing: '-0.3px' }}>
            {{ chat: '法律问答', search: '法条检索', document: '文档管理' }[activeMenu]}
          </Typography.Text>

          <Space size="middle">
            <Badge status="success" text={<span style={{ color: '#64748b', fontSize: 13 }}>系统就绪</span>} />
          </Space>
        </Header>

        <Content
          style={{
            margin: 24,
            minHeight: 280,
            overflow: 'auto',
          }}
        >
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  )
}
