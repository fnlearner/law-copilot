/**
 * LawCopilot API 服务层
 * 统一封装后端接口调用
 */

import axios, { AxiosResponse } from 'axios'
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

// 响应拦截器 — 返回 data 而非整个 AxiosResponse
api.interceptors.response.use(
  (response: AxiosResponse) => response.data,
  (error) => {
    const msg = error.response?.data?.detail || error.message || '网络请求失败'
    message.error(msg)
    return Promise.reject(error)
  },
)

// ===== 类型定义 =====

export interface AskQuestionResponse {
  reply: string
  references: Array<{
    title: string
    doc_type: string
    relevance_score: number
    content_snippet: string
  }>
  latency_ms: number
}

export interface SearchResult {
  title: string
  doc_type: string
  content: string
  relevance_score: number
  source?: string
}

export interface SearchDocumentsResponse {
  results: SearchResult[]
  latency_ms: number
}

export interface SearchSuggestionsResponse {
  suggestions: string[]
}

export interface SearchStatsResponse {
  status: string
  vectors_count: number
}

export interface DocListResponse {
  collection_info?: {
    collection: string
    vectors_count: number
    vector_size: number
    distance: string
  }
  collection?: string
  vectors_count?: number
  vector_size?: number
  distance?: string
}

export interface SeedDemoResponse {
  message: string
}

export interface ImportLawsResponse {
  data?: {
    success: number
    failed: number
  }
}

// ===== 对话接口 =====

export async function askQuestion(params: {
  message: string
  task_type: string
  scope: string
  top_k: number
}): Promise<AskQuestionResponse> {
  return api.post('/chat/ask', params)
}

// ===== 搜索接口 =====

export async function searchDocuments(params: {
  query: string
  scope: string
  top_k: number
}): Promise<SearchDocumentsResponse> {
  return api.post('/search/query', params)
}

export async function getSearchSuggestions(q: string, limit = 8): Promise<SearchSuggestionsResponse> {
  return api.get('/search/suggestions', { params: { q, limit } })
}

export async function getSearchStats(): Promise<SearchStatsResponse> {
  return api.get('/search/stats')
}

// ===== 文档管理接口 =====

export async function uploadDocument(formData: FormData) {
  return api.post('/document/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export async function importLawLibrary(): Promise<ImportLawsResponse> {
  return api.post('/document/import-laws')
}

export async function seedDemoData(): Promise<SeedDemoResponse> {
  return api.post('/document/seed-demo')
}

export async function listDocuments(): Promise<DocListResponse> {
  return api.get('/document/list')
}

// ===== 系统接口 =====

export async function healthCheck() {
  return api.get('/health')
}
