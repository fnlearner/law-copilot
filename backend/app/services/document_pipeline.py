"""
法律文档清洗 Pipeline

支持多格式法律文档的完整处理链：
  原始文件 → 格式检测 → 文本提取 → 清洗预处理 → 结构解析 → 分块 → 元数据注入 → 向量化入库
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class CleanedDocument:
    """清洗后的文档"""
    raw_text: str                    # 原始文本
    cleaned_text: str                # 清洗后文本
    metadata: Dict[str, Any] = field(default_factory=dict)   # 提取的元数据


@dataclass
class LegalChunk:
    """法条级分块"""
    content: str                     # 条文内容
    article_number: Optional[str]    # 条号（如"第一条"、"第23条"）
    article_index: int = 0           # 条序号（数字）
    chapter: Optional[str] = None    # 所属章节
    section: Optional[str] = None    # 所属节
    law_name: Optional[str] = None   # 法律名称
    doc_type: str = "law"
    source_file: str = ""
    char_offset_start: int = 0
    char_offset_end: int = 0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 1. 多格式文本提取器
# ============================================================

class TextExtractor:
    """从各种文件格式中提取纯文本"""

    @staticmethod
    def extract_from_txt(file_path: str, encoding: str = "utf-8") -> str:
        """提取 TXT 纯文本"""
        encodings = [encoding, "utf-8", "gbk", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read()
                logger.debug(f"[TXT] {os.path.basename(file_path)} → {len(text)} chars (encoding={enc})")
                return text
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"无法解码文件: {file_path}")

    @staticmethod
    def extract_from_md(file_path: str) -> str:
        """提取 Markdown 文本，保留标题结构标记"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.debug(f"[MD] {os.path.basename(file_path)} → {len(text)} chars")
        return text

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """从 PDF 提取文本"""
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            texts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
            text = "\n".join(texts)
            logger.debug(f"[PDF] {os.path.basename(file_path)} → {len(text)} chars ({len(reader.pages)} pages)")
            return text
        except ImportError:
            raise ImportError("PDF 处理需要安装 pypdf: pip install pypdf")
        except Exception as e:
            logger.error(f"[PDF] 提取失败: {e}")
            return ""

    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """从 Word (.docx) 提取文本"""
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            logger.debug(f"[DOCX] {os.path.basename(file_path)} → {len(text)} chars")
            return text
        except ImportError:
            raise ImportError("DOCX 处理需要安装 docx2txt: pip install docx2txt")

    @classmethod
    def auto_extract(cls, file_path: str) -> str:
        """自动根据扩展名选择提取器"""
        ext = os.path.splitext(file_path)[1].lower()

        extractors = {
            ".txt": cls.extract_from_txt,
            ".text": cls.extract_from_txt,
            ".md": cls.extract_from_md,
            ".markdown": cls.extract_from_md,
            ".pdf": cls.extract_from_pdf,
            ".docx": cls.extract_from_docx,
        }

        extractor = extractors.get(ext)
        if not extractor:
            raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {list(extractors.keys())}")

        return extractor(file_path)


# ============================================================
# 2. 法律文本清洗器
# ============================================================

class LegalTextCleaner:
    """
    法律文本专用清洗器
    
    处理目标：
    - 去除页眉页脚、水印文字
    - 统一标点符号和空格
    - 规范条文编号格式
    - 去除无意义空白行
    - 合并断行（处理 PDF 提取时的换行问题）
    """

    # 页面噪声模式（常见于 PDF 扫描件）
    PAGE_NOISE_PATTERNS = [
        r'^\s*—?\s*\d+\s*—?\s*$',              # 孤立的页码（如 "--- 42 ---"）
        r'^\s*第\s*\d+\s*页\s*$',               # "第X页"
        r'^\s*[A-Za-z]*LawCopilot[A-Za-z]*\s*$', # 水印文字
        r'^\s*www\.[\w\.]+\s*$',                # URL 行
        r'^\s*[\|\/\\\-]{5,}\s*$',             # 分隔线
        r'^\s*(机密|内部|保密|公开)\s*',         # 密级标注
        r'^\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日.*$',  # 日期行
    ]

    # 需要合并的断行模式（PDF 中常见的行内断开）
    LINE_BREAK_FIX_PATTERNS = [
        # 句子中间不应换行的位置
        (r'([^。！？；\n])\n([^\n])', r'\1\2'),       # 非句末换行 → 合并
        (r'(\d+)\n(条|款|项|章|节)', r'\1\2'),       # 数字+条/款 换行
        (r'(第)\n(\d+(?:条(?:之[一二三四五六七八九十])?)?)', r'\1\2'),  # "第\nX条" → "第X条"
        (r'([（(])\n', r'\1'),                        # 左括号前换行
        (r'\n([）)])', r'\1'),                         # 右括号后换行
    ]

    # 标点符号规范化
    PUNCTUATION_MAP = {
        ',': '，',
        '.': '。',
        '!': '！',
        '?': '？',
        ';': '；',
        ':': '：',
        '"': '"',   # 保持中文引号
        '"': '"',
        '\x27': '\xe2\x80\x99',
        '\x27': '\xe2\x80\x98',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】',
    }

    @classmethod
    def clean(cls, raw_text: str, file_ext: str = ".txt") -> CleanedDocument:
        """
        执行完整的文本清洗流程
        
        Args:
            raw_text: 原始文本
            file_ext: 文件扩展名，用于判断是否需要特殊处理
            
        Returns:
            CleanedDocument 对象
        """
        text = raw_text

        # Step 1: PDF 断行修复（最关键的一步）
        if file_ext == ".pdf":
            text = cls._fix_pdf_line_breaks(text)

        # Step 2: 页面噪声去除
        text = cls._remove_noise(text)

        # Step 3: 标点符号规范化
        text = cls._normalize_punctuation(text)

        # Step 4: 空白字符清理
        text = cls._clean_whitespace(text)

        # Step 5: 条文编号规范化
        text = cls._normalize_article_numbers(text)

        # 计算清洗统计
        original_len = len(raw_text)
        cleaned_len = len(text)
        reduction_ratio = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)

        metadata = {
            "original_length": original_len,
            "cleaned_length": cleaned_len,
            "reduction_percent": reduction_ratio,
            "cleaned_at": datetime.now().isoformat(),
        }

        logger.info(
            f"🧹 文本清洗完成: {original_len} → {cleaned_len} 字符 "
            f"(减少 {reduction_ratio}%, 去噪 {original_len - cleaned_len} 字符)"
        )

        return CleanedDocument(raw_text=raw_text, cleaned_text=text, metadata=metadata)

    @classmethod
    def _fix_pdf_line_breaks(cls, text: str) -> str:
        """修复 PDF 提取产生的异常断行"""
        for pattern, replacement in cls.LINE_BREAK_FIX_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text

    @classmethod
    def _remove_noise(cls, text: str) -> str:
        """移除页面级别的噪声行"""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            is_noise = False
            for noise_pattern in cls.PAGE_NOISE_PATTERNS:
                if re.match(noise_pattern, line.strip()):
                    is_noise = True
                    break
            if not is_noise:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """统一中英文标点为中文全角"""
        for eng_punc, cn_punc in cls.PUNCTUATION_MAP.items():
            text = text.replace(eng_punc, cn_punct)
        # 连续多个空格→单空格
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    @classmethod
    def _clean_whitespace(cls, text: str) -> str:
        """清理多余空白"""
        # 3个以上连续空行→2个空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 行首尾空白
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # 全文首尾空白
        text = text.strip()
        return text

    @classmethod
    def _normalize_article_numbers(cls, text: str) -> str:
        """规范化条文编号格式"""
        # 统一 "第 X 条" 格式（处理全角数字等）
        text = re.sub(r'第\s*[０-９]+\s*条', lambda m: m.group().replace(' ', ''), text)
        # 统一 "第X条之Y" 格式
        text = re.sub(r'第(\d+)条之([一二三四五六七八九十\d]+)', r'第\1条第\2款', text)
        return text


# ============================================================
# 3. 法律文本结构解析器（核心）
# ============================================================

class LegalTextParser:
    """
    法律文本结构解析器
    
    将非结构化的法律文本解析为结构化的章节-条文层级。
    
    解析能力：
      - 识别 章/节/条/款/项 层级
      - 提取条文编号和内容
      - 识别附则、附录等特殊部分
      - 自动推断法律名称
    """

    # ===== 条文编号正则（从严格到宽松）=====
    ARTICLE_PATTERN_STRICT = re.compile(
        r'^(第[一二三四五六七八九十百零\d]+条(?:之[一二三四五六七八九十\d]+)?)(?![\w])',
        re.MULTILINE
    )
    # 兼容带括号的条文编号
    ARTICLE_PATTERN_BRACKET = re.compile(
        r'^[（(]\s*第?\s*([一二三四五六七八九十百零\d]+)\s*条\s?[）)]',
        re.MULTILINE
    )
    # 宽松匹配：仅数字编号（如 "1." 或 "1、"）
    ARTICLE_PATTERN_LOOSE = re.compile(
        r'^(\d+)[\.、．\s](?=\S)',
        re.MULTILINE
    )

    # 章节标题模式
    CHAPTER_PATTERN = re.compile(
        r'^(第[一二三四五六七八九十百零\d]+[章编])(?:\s+[^\n]{2,40})?',
        re.MULTILINE
    )
    SECTION_PATTERN = re.compile(
        r'^(第[一二三四五六七八九十\d]+节)\s+[^\n]{2,40}',
        re.MULTILINE
    )

    # 法律名称识别（通常在文件开头）
    LAW_NAME_PATTERNS = [
        re.compile(r'^《(.{2,30}(?:法|条例|规定|办法|解释|决定|规则|意见|细则|通知))》'),
        re.compile(r'^([\u4e00-\u9fa5]{2,20})(?:法|条例|规定|办法)'),
    ]

    @classmethod
    def parse(cls, text: str, filename: str = "", metadata: Dict[str, Any] = None) -> List[LegalChunk]:
        """
        将法律文本解析为法条级分块列表
        
        Args:
            text: 清洗后的法律文本
            filename: 源文件名（用于推断类型）
            metadata: 已知的元数据（可覆盖自动推断）
            
        Returns:
            List[LegalChunk]: 法条分块列表
        """
        meta = metadata or {}
        chunks = []

        # Step 0: 推断法律名称
        law_name = meta.get("title") or cls._infer_law_name(text, filename)
        doc_type = meta.get("doc_type") or cls._detect_doc_type(filename)

        # Step 1: 按条文分割
        article_positions = cls._find_all_articles(text)

        if not article_positions:
            # 无明确条文编号 → 回退到段落分割
            logger.warning("未检测到标准条文格式，回退到段落分割")
            chunks = cls._fallback_chunking(text, law_name, doc_type, filename, meta)
            return chunks

        # Step 2: 为每条创建分块 + 注入章节信息
        for i, (start_pos, end_pos, article_number, article_idx) in enumerate(article_positions):
            content = text[start_pos:end_pos].strip()

            if len(content) < 4:  # 过滤太短的片段
                continue

            # 推断所属章节
            chapter, section = cls._find_containing_chapter_section(text, start_pos)

            chunk = LegalChunk(
                content=content,
                article_number=article_number,
                article_index=article_idx,
                chapter=chapter,
                section=section,
                law_name=law_name,
                doc_type=doc_type,
                source_file=filename,
                char_offset_start=start_pos,
                char_offset_end=end_pos,
                extra_metadata={
                    "chunk_id": f"{law_name}_{article_number}" if law_name else f"art_{i}",
                    "has_article_header": bool(re.match(r'^第.+条', content)),
                },
            )
            chunks.append(chunk)

        logger.info(f"📜 文本解析完成: 共 {len(chunks)} 个法条分块 | 法律: {law_name}")
        return chunks

    @classmethod
    def _find_all_articles(cls, text: str) -> List[Tuple[int, int, str, int]]:
        """
        找出所有条文的位置
        
        Returns:
            List of (start, end, article_number_string, index)
        """
        positions = []
        
        # 使用严格模式找所有条文起始位置
        matches = list(cls.ARTICLE_PATTERN_STRICT.finditer(text))
        
        if not matches:
            return positions

        for idx, match in enumerate(matches):
            start = match.start()
            art_num = match.group(1).strip()  # 如 "第一条"
            
            # 下一个条文的起始位置（或文本末尾）
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            
            positions.append((start, end, art_num, idx))

        return positions

    @classmethod
    def _find_containing_chapter_section(cls, text: str, position: int) -> Tuple[Optional[str], Optional[str]]:
        """
        给定文本位置，查找其所属的章和节
        返回 position 之前最近的章/节标题
        """
        chapter = None
        section = None

        # 查找该位置之前的所有章节标题
        text_before = text[:position]

        # 找最近的"章"
        chapter_matches = list(cls.CHAPTER_PATTERN.finditer(text_before))
        if chapter_matches:
            last_chapter = chapter_matches[-1]
            chapter = last_chapter.group().strip()

        # 找最近的"节"
        section_matches = list(cls.SECTION_PATTERN.finditer(text_before))
        if section_matches:
            last_section = section_matches[-1]
            section = last_section.group().strip()

        return chapter, section

    @classmethod
    def _infer_law_name(cls, text: str, filename: str) -> Optional[str]:
        """从文本开头或文件名推断法律名称"""
        # 先尝试从文本开头提取
        first_lines = '\n'.join(text.split('\n')[:10])
        for pattern in cls.LAW_NAME_PATTERNS:
            match = pattern.search(first_lines)
            if match:
                name = match.group(1) or match.group(0).strip()
                return name

        # 再尝试从文件名推断
        name_without_ext = os.path.splitext(os.path.basename(filename))[0]
        if any(kw in name_without_ext for kw in ["法", "条例", "规定", "办法", "解释"]):
            return name_without_ext

        return name_without_ext or "未知法规"

    @classmethod
    def _detect_doc_type(cls, filename: str) -> str:
        """根据文件名/内容特征判断文档类型"""
        name_lower = filename.lower()
        
        case_indicators = ["案号", "判决书", "裁定书", "(202", "民初", "刑初"]
        agreement_indicators = ["合同", "协议", "契约"]

        for kw in case_indicators:
            if kw in filename or kw in name_lower:
                return "case"
        for kw in agreement_indicators:
            if kw in filename or kw in name_lower:
                return "agreement"

        return "law"

    @classmethod
    def _fallback_chunking(
        cls, 
        text: str, 
        law_name: str, 
        doc_type: str, 
        filename: str, 
        meta: Dict[str, Any],
        chunk_size: int = 512,
        overlap: int = 80
    ) -> List[LegalChunk]:
        """
        回退方案：无法按条文分割时，按语义段落分块
        
        用于：
          - 判例文书（无标准条文编号）
          - 合同文本（按条款自然分段）
          - 其他非标准法律文本
        """
        # 按双空行或明显段落分隔符切分
        paragraphs = re.split(r'\n{2,}|(?<=。)\n(?=[^　\s])', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

        chunks = []
        current_content = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current_content) + len(para) > chunk_size and current_content:
                # 当前缓冲区已满 → 输出一个分块
                chunks.append(LegalChunk(
                    content=current_content.strip(),
                    article_number=None,
                    article_index=chunk_idx,
                    chapter=meta.get("chapter"),
                    law_name=law_name,
                    doc_type=doc_type,
                    source_file=filename,
                    extra_metadata={"chunk_method": "paragraph_fallback"},
                ))
                chunk_idx += 1
                # 保留重叠部分作为上下文衔接
                current_content = current_content[-overlap:] + "\n\n" + para
            else:
                current_content += "\n\n" + para if current_content else para

        # 最后剩余的内容
        if current_content.strip():
            chunks.append(LegalChunk(
                content=current_content.strip(),
                article_number=None,
                article_index=chunk_idx,
                chapter=meta.get("chapter"),
                law_name=law_name,
                doc_type=doc_type,
                source_file=filename,
                extra_metadata={"chunk_method": "paragraph_fallback"},
            ))

        return chunks


# ============================================================
# 4. Pipeline 组装器
# ============================================================

class DocumentProcessingPipeline:
    """
    完整的文档处理流水线
    
    流程链：
      file_path → TextExtractor → LegalTextCleaner → LegalTextParser → List[LegalChunk]
    """

    def __init__(self):
        self.extractor = TextExtractor()
        self.cleaner = LegalTextCleaner()
        self.parser = LegalTextParser()

    async def process_file(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        处理单个文件的完整流水线
        
        Returns:
            {
                "filename": str,
                "status": "ok" | "error",
                "raw_length": int,
                "cleaned_length": int,
                "chunks_count": int,
                "chunks": List[LegalChunk],
                "metadata": Dict,
                "processing_time_ms": int,
            }
        """
        start = datetime.now()
        meta = metadata or {}
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        result = {"filename": filename, "source_file": file_path}

        try:
            # === Stage 1: 文本提取 ===
            logger.info(f"📂 Stage 1/3 提取文本: {filename}")
            raw_text = self.extractor.auto_extract(file_path)
            result["raw_length"] = len(raw_text)

            if not raw_text or len(raw_text.strip()) < 20:
                result.update({"status": "skipped", "reason": "文本过短或为空"})
                return result

            # === Stage 2: 文本清洗 ===
            logger.info(f"🧹 Stage 2/3 清洗文本: {filename}")
            cleaned_doc = self.cleaner.clean(raw_text, ext)
            result["cleaned_length"] = len(cleaned_doc.cleaned_text)
            result["clean_stats"] = cleaned_doc.metadata

            # 合并外部元数据与清洗元数据
            merged_meta = {**meta, **cleaned_doc.metadata}
            merged_meta.setdefault("title", filename)

            # === Stage 3: 结构解析 & 分块 ===
            logger.info(f"📜 Stage 3/3 解析分块: {filename}")
            chunks = self.parser.parse(cleaned_doc.cleaned_text, filename, merged_meta)

            # 补充额外元数据
            for chunk in chunks:
                chunk.source_file = file_path
                chunk.extra_metadata["ingested_at"] = datetime.now().isoformat()

            elapsed_ms = int((datetime.now() - start).total_seconds() * 1000)

            result.update({
                "status": "ok",
                "chunks_count": len(chunks),
                "chunks": chunks,
                "metadata": {
                    "law_name": chunks[0].law_name if chunks else filename,
                    "doc_type": chunks[0].doc_type if chunks else "unknown",
                    **merged_meta,
                },
                "processing_time_ms": elapsed_ms,
            })

            logger.info(
                f"✅ 处理完成: {filename} | "
                f"{result['raw_length']}→{result['cleaned_length']}字 | "
                f"{result['chunks_count']}分块 | {elapsed_ms}ms"
            )

            return result

        except Exception as e:
            elapsed_ms = int((datetime.now() - start).total_seconds() * 1000)
            logger.error(f"❌ 处理失败: {filename} → {e}", exc_info=True)
            result.update({
                "status": "error",
                "error_message": str(e),
                "processing_time_ms": elapsed_ms,
            })
            return result

    async def process_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.{txt,md,pdf,docx}",
        metadata_overrides: Dict[str, Any] = None,
        max_files: int = 500,
    ) -> Dict[str, Any]:
        """
        批量处理目录中的所有法律文档
        """
        import glob as glob_module

        full_pattern = os.path.join(directory_path, glob_pattern)
        files = sorted(glob_module.glob(full_pattern, recursive=True))[:max_files]

        results = {
            "total_files": len(files),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "files_detail": [],
            "all_chunks": [],
            "errors": [],
        }

        logger.info(f"📁 开始批量处理目录: {directory_path} ({len(files)} files)")

        for file_path in files:
            file_result = await self.process_file(file_path, metadata_overrides)

            results["files_detail"].append({
                "name": file_result.get("filename", ""),
                "status": file_result.get("status", "unknown"),
                "chunks": file_result.get("chunks_count", 0),
                "error": file_result.get("error_message", ""),
            })

            if file_result["status"] == "ok":
                results["success"] += 1
                results["total_chunks"] += file_result["chunks_count"]
                results["all_chunks"].extend(file_result.get("chunks", []))
            elif file_result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file": file_result.get("filename", ""),
                    "error": file_result.get("error_message", ""),
                })

        logger.info(
            f"📊 目录处理完毕: 成功={results['success']}, "
            f"失败={results['failed']}, 跳过={results['skipped']}, "
            f"总分块={results['total_chunks']}"
        )

        return results
