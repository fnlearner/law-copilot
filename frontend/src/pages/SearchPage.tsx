import React, { useState, useEffect } from 'react'
import {
  Input,
  Button,
  Select,
  Card,
  Tag,
  Space,
  Typography,
  Empty,
  Spin,
  Row,
  Col,
  Progress,
  Tooltip,
  message,
} from 'antd'
import {
  SearchOutlined,
  FileTextOutlined,
  BookOutlined,
  DatabaseOutlined,
  FilterOutlined,
  ClearOutlined,
} from '@ant-design/icons'
import { searchDocuments, getSearchSuggestions, getSearchStats } from '../services/api'

const { Text, Paragraph, Title } = Typography
const { Search } = Input

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState('searchScope.ALL')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const [stats, setStats] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)
  const [latency, setLatency] = useState(0)

  useEffect(() => {
    loadStats()
  }, [])

  // 加载向量库统计信息
  const loadStats = async () => {
    try {
      const data = await getSearchStats()
      setStats(data)
    } catch (e) {
      console.log('Stats not available')
    }
  }

  // 输入变化时获取搜索建议
  const handleInputChange = (value) => {
    setQuery(value)
    if (value.length >= 1) {
      getSearchSuggestions(value).then((data) => {
        setSuggestions(data.suggestions || [])
      }).catch(() => {})
    }
  }

  // 执行搜索
  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setHasSearched(true)

    try {
      const data = await searchDocuments({
        query: query.trim(),
        scope: scope,
        top_k: 10,
      })

      setResults(data.results || [])
      setLatency(data.latency_ms || 0)

      if ((data.results || []).length === 0) {
        message.info('未找到匹配结果，建议更换关键词或扩大检索范围')
      }
    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setLoading(false)
    }
  }

  const getDocTypeIcon = (type) => {
    switch (type) {
      case 'law': return <BookOutlined style={{ color: '#3b82f6' }} />
      case 'case': return <FileTextOutlined style={{ color: '#f59e0b' }} />
      case 'judicial': return <DatabaseOutlined style={{ color: '#8b5cf6' }} />
      default: return <FileTextOutlined style={{ color: '#64748b' }} />
    }
  }

  const getDocTypeTagColor = (type) => {
    switch (type) {
      case 'law': return 'blue'
      case 'case': return 'gold'
      case 'judicial': return 'purple'
      default: return 'default'
    }
  }

  return (
    <div className="page-enter">
      {/* 搜索栏 */}
      <Card
        style={{
          marginBottom: 20,
          borderRadius: 14,
          border: '1px solid #e2e8f0',
          background: 'linear-gradient(135deg, #fff 0%, #f8fafc 100%)',
        }}
      >
        <Row gutter={[16, 12]} align="middle">
          <Col flex="auto">
            <Search
              value={query}
              onChange={(e) => handleInputChange(e.target.value)}
              onSearch={handleSearch}
              placeholder="输入法律关键词进行检索（例如：股权转让、合同解除、内幕交易...）"
              enterButton={
                <Button type="primary" icon={<SearchOutlined />} size="large" loading={loading}>
                  检索法条
                </Button>
              }
              size="large"
              allowClear
              style={{ borderRadius: 10 }}
            />
          </Col>
          <Col>
            <Select
              value={scope}
              onChange={setScope}
              style={{ width: 160 }}
              size="large"
              options={[
                { value: 'searchScope.ALL', label: '全部类型' },
                { value: 'searchScope.LAWS', label: '仅法律法规' },
                { value: 'searchScope.CASES', label: '仅判例案例' },
                { value: 'searchScope.ECONOMIC', label: '经济类优先' },
              ]}
            />
          </Col>
        </Row>

        {/* 搜索建议 */}
        {suggestions.length > 0 && !hasSearched && (
          <div style={{ marginTop: 10 }}>
            <Space wrap size={[4, 6]}>
              {suggestions.map((s, i) => (
                <Tag
                  key={i}
                  style={{ cursor: 'pointer', borderRadius: 14, padding: '2px 10px' }}
                  onClick={() => { setQuery(s); setSuggestions([]); }}
                  color="processing"
                >
                  {s}
                </Tag>
              ))}
            </Space>
          </div>
        )}

        {/* 统计信息 */}
        {stats && stats.status === 'ready' && (
          <div style={{ marginTop: 14 }}>
            <Space size="large">
              <Text type="secondary" style={{ fontSize: 13 }}>
                📊 向量库已收录 <Text strong>{stats.vectors_count}</Text> 条法规片段
              </Text>
              {latency > 0 && (
                <Text type="secondary" style={{ fontSize: 13 }}>
                  ⚡ 耗时 {latency}ms
                </Text>
              )}
            </Space>
          </div>
        )}
      </Card>

      {/* 搜索结果 */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <Spin size="large" tip="正在检索相关法律法规..." />
        </div>
      ) : hasSearched ? (
        results.length > 0 ? (
          <Row gutter={[16, 16]}>
            {results.map((item, idx) => (
              <Col xs={24} key={idx}>
                <Card
                  className="search-result-item"
                  size="small"
                  hoverable
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                    {/* 左侧序号 + 图标 */}
                    <div style={{
                      minWidth: 36,
                      height: 36,
                      borderRadius: 8,
                      background: '#eff6ff',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}>
                      {getDocTypeIcon(item.doc_type)}
                    </div>

                    {/* 中间内容 */}
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <Title level={5} style={{ margin: 0, fontSize: 15 }}>{item.title}</Title>
                        <Tag color={getDocTypeTagColor(item.doc_type)} style={{ borderRadius: 10 }}>
                          {{ law: '法规', case: '案例', judicial: '司法解释', other: '其他' }[item.doc_type] || '文档'}
                        </Tag>
                      </div>

                      <Paragraph
                        ellipsis={{ rows: 3, expandable: true, symbol: '展开全文' }}
                        style={{ color: '#475569', lineHeight: 1.7, fontSize: 13, marginBottom: 8 }}
                      >
                        {item.content}
                      </Paragraph>

                      {/* 底部元数据 */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Space size={12}>
                          <Tooltip title="相关度评分">
                            <Progress
                              percent={Math.round(item.relevance_score * 100)}
                              size="small"
                              status={item.relevance_score > 0.7 ? 'success' : item.relevance_score > 0.4 ? 'normal' : 'exception'}
                              style={{ width: 120 }}
                            />
                          </Tooltip>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            来源：{item.source || '-'}
                          </Text>
                        </Space>
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty
            description={<span>未找到匹配的法规内容<br/><span style={{ color: '#94a3b8', fontSize: 12 }}>尝试使用更通用的关键词</span></span>}
            style={{ padding: 60 }}
          />
        )
      ) : (
        /* 初始空状态 */
        <Empty
          description={
            <div>
              <p style={{ fontSize: 15 }}>输入关键词开始检索法律条文和案例</p>
              <p style={{ color: '#94a3b8', fontSize: 12 }}>支持语义检索，无需精确匹配法条原文</p>
            </div>
          }
          style={{ padding: 80 }}
        >
          {/* 热门搜索词 */}
          <div style={{ marginTop: 16 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>热门搜索：</Text>
            <Space wrap size={[4, 6]} style={{ marginLeft: 8 }}>
              {['股东责任', '违约赔偿', '破产重整', '竞业限制', '内幕交易', '垄断协议'].map((kw, i) => (
                <Tag
                  key={i}
                  color="blue"
                  style={{ cursor: 'pointer', borderRadius: 14 }}
                  onClick={() => { setQuery(kw); handleSearch(); }}
                >
                  {kw}
                </Tag>
              ))}
            </Space>
          </div>
        </Empty>
      )}
    </div>
  )
}


