import React, { useState, useRef, useEffect } from 'react'
import {
  Input,
  Button,
  Select,
  Tag,
  Tooltip,
  Space,
  Typography,
  Card,
  Empty,
  Spin,
  message,
} from 'antd'
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  CopyOutlined,
  FileSearchOutlined,
  ClearOutlined,
  LoadingOutlined,
} from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import { askQuestion } from '../services/api'

const { TextArea } = Input
const { Text, Paragraph } = Typography

// 预设问题快捷入口
const quickQuestions = [
  '有限责任公司股东转让股权需要什么程序？',
  '合同违约后如何计算损失赔偿额？',
  '内幕交易的认定标准和法律责任是什么？',
  '竞业限制协议的效力如何判断？',
  '经营者集中申报的门槛是什么？',
]

interface Message {
  role: 'user' | 'assistant'
  content: string
  references?: Array<{
    title: string
    doc_type: string
    relevance_score: number
    content_snippet: string
  }>
  latency_ms?: number
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `您好，我是 **LawCopilot** 法律研究助手。👨‍⚖️\n\n我可以帮您：\n- 🔍 **检索法条** — 快速找到相关法律法规条文\n- 📋 **分析案例** — 基于判例提供类案参考\n- 📝 **生成文书** — 辅助起草法律文书\n- 🎯 **法律研究** — 各类法律法规的深度研究与分析\n\n请输入您的法律问题，我会基于法律法规和案例为您进行分析。`,
    },
  ])
  const [inputValue, setInputValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [taskType, setTaskType] = useState('legal_research')
  const messagesEndRef = useRef(null)

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const response = await askQuestion({
        message: userMessage,
        task_type: taskType,
        scope: 'economic',
        top_k: 5,
      })

      const aiMessage: Message = {
        role: 'assistant' as const,
        content: response.reply || '抱歉，未能生成回复。',
        references: response.references || [],
        latency_ms: response.latency_ms,
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ 请求失败：${error.message || '网络错误'}。请检查后端服务是否正常运行。`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text)
    message.success('已复制到剪贴板')
  }

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: '对话已清空。请问有什么法律问题需要我帮助？',
      },
    ])
  }

  return (
    <div className="page-enter" style={{ height: '100%' }}>
      {/* 快捷问题 + 工具栏 */}
      <Card
        size="small"
        style={{
          marginBottom: 16,
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          background: '#fff',
        }}
      >
        <Space wrap>
          <span style={{ color: '#64748b', fontSize: 13, marginRight: 4 }}>任务类型：</span>
          <Select
            value={taskType}
            onChange={setTaskType}
            style={{ width: 160 }}
            size="middle"
            options={[
              { value: 'legal_research', label: '🔍 法律研究' },
              { value: 'search_law', label: '📕 法条检索' },
              { value: 'analyze_case', label: '📋 案例分析' },
              { value: 'generate_doc', label: '📝 文书生成' },
            ]}
          />

          <Button
            icon={<ClearOutlined />}
            size="middle"
            onClick={clearChat}
            type="text"
            danger
          >
            清空对话
          </Button>
        </Space>

        <div style={{ marginTop: 10 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>常见问题：</Text>
          <Space size={[4, 8]} wrap style={{ marginLeft: 6 }}>
            {quickQuestions.map((q, i) => (
              <Tag
                key={i}
                color="blue"
                style={{ cursor: 'pointer', borderRadius: 14, padding: '2px 10px' }}
                onClick={() => {
                  setInputValue(q)
                  handleSend()
                }}
              >
                {q.length > 20 ? q.slice(0, 20) + '...' : q}
              </Tag>
            ))}
          </Space>
        </div>
      </Card>

      {/* 聊天主区域 */}
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
              <div className={`message-bubble message-${msg.role}`}>
                {msg.role === 'assistant' && (
                  <div style={{ marginBottom: 6 }}>
                    <RobotOutlined style={{ marginRight: 6, color: '#3b82f6' }} />
                    <Text strong style={{ color: '#3b82f6', fontSize: 13 }}>LawCopilot</Text>
                    {msg.latency_ms && (
                      <Text type="secondary" style={{ fontSize: 11, marginLeft: 8 }}>
                        {msg.latency_ms}ms
                      </Text>
                    )}
                  </div>
                )}
                {msg.role === 'user' && (
                  <div style={{ marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <UserOutlined /> <Text style={{ fontSize: 13, opacity: 0.9 }}>律师</Text>
                  </div>
                )}

                {/* AI 回复用 Markdown 渲染 */}
                <div className={`ai-response ${msg.role}`}>
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>

                {/* 引用来源 */}
                {msg.references && msg.references.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <Text strong style={{ fontSize: 12, color: '#475569' }}>
                      <FileSearchOutlined style={{ marginRight: 4 }} />
                      引用来源 ({msg.references.length})
                    </Text>
                    {msg.references.map((ref, i) => (
                      <div key={i} className="reference-card">
                        <Space direction="vertical" size={2}>
                          <Text strong style={{ fontSize: 13 }}>{ref.title}</Text>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            类型：{ref.doc_type} | 相关度：{(ref.relevance_score * 100).toFixed(1)}%
                          </Text>
                          <Paragraph
                            ellipsis={{ rows: 2 }}
                            style={{ fontSize: 12, marginBottom: 0, color: '#475569' }}
                          >
                            {ref.content_snippet}
                          </Paragraph>
                        </Space>
                      </div>
                    ))}
                  </div>
                )}

                {/* 复制按钮 */}
                <div style={{ textAlign: 'right', marginTop: 8 }}>
                  <Tooltip title="复制回复内容">
                    <Button
                      type="text"
                      icon={<CopyOutlined />}
                      size="small"
                      onClick={() => handleCopy(msg.content)}
                      style={{ color: '#94a3b8' }}
                    />
                  </Tooltip>
                </div>
              </div>
            </div>
          ))}

          {/* 加载中指示器 */}
          {loading && (
            <div className="message-bubble message-ai">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <RobotOutlined style={{ color: '#3b82f6' }} />
                <Text type="secondary">正在检索法规并分析...</Text>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* 输入区域 */}
        <div className="chat-input-area">
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="请描述您的法律问题...（例如：股东未履行出资义务，其他股东可以怎么做？）"
            autoSize={{ minRows: 2, maxRows: 5 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            style={{
              borderRadius: 12,
              border: '1px solid #e2e8f0',
              resize: 'none',
              padding: '12px 16px',
              fontSize: 14,
              lineHeight: 1.6,
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>Enter 发送 · Shift+Enter 换行</Text>
            <Button
              type="primary"
              icon={loading ? <LoadingOutlined /> : <SendOutlined />}
              onClick={handleSend}
              loading={loading}
              disabled={!inputValue.trim()}
              style={{
                borderRadius: 20,
                height: 40,
                paddingLeft: 24,
                paddingRight: 24,
                fontWeight: 500,
                background: 'linear-gradient(135deg, #1a56db 0%, #2563eb 100%)',
                border: 'none',
              }}
            >
              发送提问
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
