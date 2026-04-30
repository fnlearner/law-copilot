import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Upload,
  Select,
  Table,
  Tag,
  Space,
  Typography,
  Modal,
  message,
  Row,
  Col,
  Statistic,
  Alert,
  Steps,
  Divider,
} from 'antd'
import {
  UploadOutlined,
  FileTextOutlined,
  InboxOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import {
  uploadDocument,
  importLawLibrary,
  seedDemoData,
  listDocuments,
} from '../services/api'

const { Text, Title, Paragraph } = Typography
const { Dragger } = Upload

export default function DocumentPage() {
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [docList, setDocList] = useState([])
  const [stats, setStats] = useState(null)
  const [seedModalOpen, setSeedModalOpen] = useState(false)

  useEffect(() => {
    loadDocInfo()
  }, [])

  // 加载文档列表信息
  const loadDocInfo = async () => {
    try {
      const data = await listDocuments()
      setStats(data.collection_info || data)
    } catch (e) {
      console.log('Doc list error:', e)
    }
  }

  // 文件上传
  const handleUpload = async (file) => {
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('title', file.name.replace(/\.[^/.]+$/, ''))
      formData.append('doc_type', 'other')
      formData.append('category', '')
      formData.append('description', '')

      const res = await uploadDocument(formData)

      message.success(`${file.name} 上传成功，已完成向量化`)
      loadDocInfo()

      return false // 阻止默认上传行为
    } catch (error) {
      message.error(`上传失败: ${error.message}`)
      return false
    } finally {
      setUploading(false)
    }
  }

  // 导入法律库目录
  const handleImportLaws = async () => {
    setLoading(true)

    Modal.confirm({
      title: '导入法律数据',
      icon: <InfoCircleOutlined />,
      content: (
        <div>
          <p>将从 <Text code>./data/laws</Text> 目录批量导入法律文本文件。</p>
          <p style={{ color: '#f59e0b' }}>请确保法律文件已放置在正确位置。</p>
        </div>
      ),
      okText: '开始导入',
      cancelText: '取消',
      onOk: async () => {
        try {
          const res = await importLawLibrary()
          Modal.success({
            title: '导入完成',
            content: (
              <div>
                <p>成功导入: {res.data?.success || 0} 个文件</p>
                <p>失败: {res.data?.failed || 0} 个文件</p>
              </div>
            ),
          })
          loadDocInfo()
        } catch (error) {
          message.error(`导入失败: ${error.message}`)
        } finally {
          setLoading(false)
        }
      },
      onCancel: () => setLoading(false),
    })
  }

  // 种子数据（演示用）
  const handleSeedDemo = async () => {
    setSeedModalOpen(true)

    try {
      const res = await seedDemoData()
      message.success(res.message)
      setSeedModalOpen(false)
      loadDocInfo()
    } catch (error) {
      message.error(error.response?.data?.detail || '种子数据写入失败')
      setSeedModalOpen(false)
    }
  }

  return (
    <div className="page-enter">
      {/* 操作面板 */}
      <Row gutter={[20, 16]}>
        {/* 上传文档 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <CloudUploadOutlined />
                <span>上传法律文档</span>
              </Space>
            }
            size="small"
            style={{ borderRadius: 14, height: '100%' }}
          >
            <Dragger
              name="file"
              multiple={false}
              accept=".pdf,.docx,.md,.txt"
              showUploadList={false}
              beforeUpload={(file) => handleUpload(file)}
              disabled={uploading}
              className="upload-dragger"
              style={{
                borderRadius: 12,
                padding: '30px 20px',
                background: '#f8fafc',
                borderStyle: uploading ? 'solid' : 'dashed',
                borderColor: uploading ? '#3b82f6' : '#bfdbfe',
              }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined style={{ fontSize: 40, color: '#3b82f6' }} />
              </p>
              <p className="ant-upload-text" style={{ fontWeight: 500 }}>
                点击或拖拽上传法律文档
              </p>
              <p className="ant-upload-hint" style={{ fontSize: 12, color: '#94a3b8' }}>
                支持 PDF / Word(.docx) / Markdown / 纯文本 · 上传后自动向量化入库
              </p>
            </Dragger>

            {uploading && (
              <div style={{ textAlign: 'center', marginTop: 12 }}>
                <LoadingOutlined spin /> <span style={{ marginLeft: 8 }}>正在处理并向量...</span>
              </div>
            )}
          </Card>
        </Col>

        {/* 数据管理 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <DatabaseOutlined />
                <span>数据管理</span>
              </Space>
            }
            extra={<Text type="secondary" style={{ fontSize: 12 }}>一键操作</Text>}
            size="small"
            style={{ borderRadius: 14 }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {/* 种子数据 */}
              <Button
                type="primary"
                icon={<DatabaseOutlined />}
                onClick={handleSeedDemo}
                block
                size="large"
                style={{ borderRadius: 10, height: 48, background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)' }}
              >
                📥 写入示范法条（演示数据）
              </Button>

              <Paragraph type="secondary" style={{ fontSize: 11, margin: 0 }}>
                写入公司法、民法典合同编、证券法、劳动争议解释、反垄断法等经济类核心法规的示范性条文。
                无需外部文件即可体验完整功能。
              </Paragraph>

              <Divider style={{ margin: '4px 0' }} />

              {/* 批量导入 */}
              <Button
                icon={<InboxOutlined />}
                onClick={handleImportLaws}
                loading={loading}
                block
                size="large"
                style={{ borderRadius: 10 }}
              >
                从 LawRefBook 目录导入
              </Button>

              <Paragraph type="secondary" style={{ fontSize: 11, margin: 0 }}>
                从 GitHub: LawRefBook/Laws 克隆的法律文本目录进行批量导入。
                需要预先将法律文件放入 ./data/laws 目录。
              </Paragraph>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 向量库状态 */}
      <Card
        title={<Space><FileTextOutlined /><span>向量数据库状态</span></Space>}
        style={{ marginTop: 20, borderRadius: 14 }}
        size="small"
      >
        {stats ? (
          <Row gutter={[24, 16]}>
            <Col xs={12} sm={6}>
              <Statistic
                title="已收录片段"
                value={stats.vectors_count || 0}
                prefix={<FileTextOutlined />}
                valueStyle={{ color: stats.vectors_count > 0 ? '#3b82f6' : '#94a3b8' }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="集合名称"
                value={stats.collection || '-'}
                prefix={<DatabaseOutlined />}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="向量维度"
                value={stats.vector_size || '-'}
                suffix="d"
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="距离算法"
                value={stats.distance === 'Cosine' ? '余弦相似度' : (stats.distance || '-')}
              />
            </Col>
          </Row>
        ) : (
          <Alert
            description={
              <div style={{ textAlign: 'center', padding: 10 }}>
                <InboxOutlined style={{ fontSize: 32, color: '#cbd5e1', marginBottom: 8 }} />
                <p>向量库尚未初始化或无法连接</p>
                <p style={{ fontSize: 12, color: '#94a3b8' }}>点击上方「写入示范法条」按钮初始化数据</p>
              </div>
            }
            type="info"
            showIcon={false}
          />
        )}
      </Card>

      {/* 使用指引 */}
      <Card
        style={{ marginTop: 20, borderRadius: 14 }}
        size="small"
        type="inner"
        title="📖 使用指引"
      >
        <Steps
          direction="horizontal"
          current={-1}
          size="small"
          items={[
            {
              title: '准备数据',
              description: '从 GitHub 克隆 Laws 仓库到 data/laws 目录',
            },
            {
              title: '配置 API Key',
              description: '在 backend/.env 填写 DeepSeek API Key',
            },
            {
              title: '启动服务',
              description: 'docker-compose up -d 一键启动全部服务',
            },
            {
              title: '开始使用',
              description: '在「法律问答」页面提问即可',
            },
          ]}
          style={{ marginBottom: 12 }}
        />

        <Alert
          type="warning"
          showIcon
          message="首次使用建议：先点击「写入示范法条」按钮，无需任何外部数据即可体验完整的 RAG 检索+LLM 分析流程。"
        />
      </Card>

      {/* 种子数据弹窗 */}
      <Modal
        open={seedModalOpen}
        title="正在写入示范法条..."
        footer={null}
        closable={false}
        centered
      >
        <div style={{ textAlign: 'center', padding: 24 }}>
          <LoadingOutlined style={{ fontSize: 36, color: '#3b82f6' }} spin />
          <p style={{ marginTop: 16, color: '#64748b' }}>正在将示范性法规向量化写入 Qdrant...</p>
          <Text type="secondary" style={{ fontSize: 12 }}>包含公司法、合同编、证券法、劳动争议解释、反垄断法等</Text>
        </div>
      </Modal>
    </div>
  )
}
