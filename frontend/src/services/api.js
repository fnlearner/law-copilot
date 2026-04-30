/**
 * LawCopilot API 服务层
 * 统一封装后端接口调用
 */

import axios from 'axios'
import { message } from 'antd'

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000, // LLM 响应可能较慢，设置较长超时
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => Promise.reject(error),
)

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg = error.response?.data?.detail || error.message || '网络请求失败'
    message.error(msg)
    return Promise.reject(error)
  },
)

// ===== 对话接口 =====

export async function askQuestion(params) {
  const res = await api.post('/chat/ask', params)
  return res
}

// ===== 搜索接口 =====

export async function searchDocuments(params) {
  const res = await api.post('/search/query', params)
  return res
}

export async function getSearchSuggestions(q, limit = 8) {
  const res = await api.get('/search/suggestions', { params: { q, limit } })
  return res
}

export async function getSearchStats() {
  const res = await api.get('/search/stats')
  return res
}

// ===== 文档管理接口 =====

export async function uploadDocument(formData) {
  const res = await api.post('/document/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res
}

export async function importLawLibrary() {
  const res = await api.post('/document/import-laws')
  return res
}

export async function seedDemoData() {
  const res = await api.post('/document/seed-demo')
  return res
}

export async function listDocuments() {
  const res = await api.get('/document/list')
  return res
}

// ===== 系统接口 =====

export async function healthCheck() {
  const res = await api.get('/health')
  return res
}
