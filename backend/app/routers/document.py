"""
文档管理接口 - 法律文档上传、处理状态、管理

v2: 集成完整 DocumentPipeline（清洗 → 解析 → 分块 → 元数据注入）
"""

import os
import uuid
import shutil
import logging
from datetime import datetime
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from typing import List

from app.models.schemas import DocumentUploadRequest, DocumentInfo, APIResponse
from app.services.rag_service import RAGService
from app.services.document_pipeline import (
    TextExtractor,
    LegalTextCleaner,
    LegalTextParser,
    DocumentProcessingPipeline,
)
from app.config import settings


logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


# ==================== 文档上传（走完整 Pipeline） ====================

@router.post("/upload", response_model=APIResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: str = Form("other"),
    category: str = Form(""),
    description: str = Form(""),
    request: Request = None,
):
    """
    上传法律文档 — 完整 Pipeline 处理
    
    流程：
      1. 保存文件到本地
      2. TextExtractor 提取文本 (TXT/MD/PDF/DOCX)
      3. LegalTextCleaner 清洗 (断行修复 / 去噪 / 标点规范化)
      4. LegalTextParser 结构解析 (按"第X条"法条级分块)
      5. 元数据注入 (条号/章/节/法律名)
      6. 向量化入库 Qdrant
    """
    rag = request.app.state.rag_service

    allowed_ext = {".pdf", ".docx", ".md", ".txt"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，允许: {allowed_ext}")

    try:
        # 保存文件
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        doc_id = str(uuid.uuid4())[:8]
        save_name = f"{doc_id}_{file.filename}"
        save_path = os.path.join(settings.UPLOAD_DIR, save_name)

        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # 构建元数据
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "doc_type": doc_type,
            "category": category or "未分类",
            "description": description,
            "original_filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
        }

        # ★ 使用完整 Pipeline 处理（核心改动）
        result = await rag.ingest_raw_file(save_path, metadata)

        if result["status"] == "ok":
            logger.info(
                f"✅ 文档上传处理完成: {file.filename} | "
                f"原始{result['raw_length']}字 → 清洗后{result['cleaned_length']}字 | "
                f"解析出 {result['chunks_count']} 个法条分块"
            )

            return APIResponse(
                code=0,
                message="文档上传成功，已完成清洗+解析+向量化",
                data={
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "title": title,
                    "doc_type": result.get("metadata", {}).get("doc_type", doc_type),
                    "law_name": result.get("metadata", {}).get("law_name"),
                    "raw_length": result.get("raw_length"),
                    "cleaned_length": result.get("cleaned_length"),
                    "chunks_parsed": result["chunks_count"],
                    "vectors_ingested": result.get("vectors_ingested", result["chunks_count"]),
                    "processing_time_ms": result.get("processing_time_ms"),
                },
            )
        else:
            return APIResponse(
                code=500,
                message=f"文档处理失败: {result.get('error', '未知错误')}",
                data={"filename": file.filename},
            )

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 批量导入法律库 ====================

@router.post("/import-laws", response_model=APIResponse)
async def import_law_library(request: Request):
    """批量导入法律目录（使用完整 Pipeline）"""
    rag = request.app.state.rag_service

    data_dir = settings.DATA_DIR
    if not os.path.exists(data_dir):
        return APIResponse(code=404, message=f"数据目录不存在: {data_dir}")

    try:
        stats = await rag.ingest_directory(data_dir)
        return APIResponse(
            code=0,
            message="法律库导入完成",
            data={**stats, "data_directory": data_dir},
        )
    except Exception as e:
        logger.error(f"Import laws error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文档列表与状态 ====================

@router.get("/list")
async def list_documents(request: Request):
    """获取向量库状态信息"""
    rag = request.app.state.rag_service
    info = await rag.get_collection_info()
    return {
        "collection_info": info,
        "data_dir": settings.DATA_DIR,
    }


# ==================== 删除文档 ====================

@router.delete("/{doc_id}", response_model=APIResponse)
async def delete_document(doc_id: str, request: Request):
    """软删除（开发中）"""
    return APIResponse(
        code=0,
        message=f"文档 {doc_id} 已标记删除（功能开发中）",
    )


# ==================== 种子数据（示范法条） ====================

@router.post("/seed-demo", response_model=APIResponse)
async def seed_demo_data(request: Request):
    """写入示范性经济类法规（通过 Pipeline 清洗解析）"""
    rag = request.app.state.rag_service

    demo_laws = [
        {
            "title": "中华人民共和国公司法（2023修订）",
            "content": """中华人民共和国公司法

第一章 总则

第一条 为了规范公司的组织和行为，保护公司、股东和债权人的合法权益，完善中国特色现代企业制度，弘扬企业家精神，维护社会经济秩序，促进社会主义市场经济的发展，根据宪法，制定本法。

第二条 本法所称公司，是指依照本法在中华人民共和国境内设立的有限责任公司和股份有限公司。

第三条 公司是企业法人，有独立的法人财产，享有法人财产权。公司以其全部财产对公司的债务承担责任。

有限责任公司的股东以其认缴的出资额为限对公司承担责任；股份有限公司的股东以其认购的股份为限对公司承担责任。

第四条 公司股东依法享有资产收益、参与重大决策和选择管理者等权利。

第二章 公司登记

第六条 设立公司，应当依法向公司登记机关申请设立登记。符合本法规定的设立条件的，由公司登记机关分别登记为有限责任公司或者股份有限公司；不符合本法规定设立条件的，不得登记为有限责任公司或者股份有限公司。

法律、行政法规规定设立公司必须报经批准的，应当在公司登记前依法办理批准手续。

公众可以向公司登记机关申请查询公司登记事项，公司登记机关应当提供查询服务。

第三章 有限责任公司的设立和组织机构

第二十三条 设立有限责任公司，应当具备下列条件：
（一）股东符合法定人数；
（二）有符合公司章程规定的全体股东认缴的出资额；
（三）股东共同制定公司章程；
（四）有公司名称，建立符合有限责任公司要求的组织机构；
（五）有公司住所。

第二十四条 有限责任公司由五十个以下股东出资设立。

第二十五条 有限责任公司章程应当载明下列事项：
（一）公司名称和住所；
（二）公司经营范围；
（三）公司注册资本；
（四）股东的姓名或者名称；
（五）股东的出资方式、出资额和出资时间；
（六）公司的机构及其产生办法、职权、议事规则；
（七）公司法定代表人；
（八）股东会会议认为需要规定的其他事项。

股东应当在公司章程上签名、盖章。

第二十六条 有限责任公司的注册资本为在公司登记机关登记的全体股东认缴的出资额。

法律、行政法规以及国务院决定对有限责任公司注册资本实缴、注册资本最低限额、股东出资期限另有规定的，从其规定。

第二十七条 股东可以用货币出资，也可以用实物、知识产权、土地使用权等可以用货币估价并可以依法转让的非货币财产作价出资；但是，法律、行政法规规定不得作为出资的财产除外。

对作为出资的非货币财产应当评估作价，核实财产，不得高估或者低估作价。法律、行政法规对评估作价有规定的，从其规定。

第八十八条 股东转让股权的，应当将股权转让的数量、价格、支付方式和期限等事项书面通知其他股东，其他股东在同等条件下有优先购买权。
""",
            "metadata": {
                "doc_type": "law",
                "category": "商法-公司法",
                "source_code": "公司法_2023修订",
                "effective_date": "2024-07-01",
            },
        },
        {
            "title": "中华人民共和国民法典 合同编（摘录）",
            "content": """中华人民共和国民法典 第三编 合同

第一分编 通则
第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。

第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。
婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定；没有规定的，可以根据其性质参照适用本编的规定。

第四百六十五条 依法成立的合同，受法律保护。
依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。

第四百六十六条 当事人对合同条款的理解有争议的，应当依据本法第一百四十二条第一款的规定，确定争议条款的含义。
合同文本采用两种以上文字订立并约定具有同等效力的，对各文本使用的词句推定具有相同含义。各文本使用的词句不一致的，应当根据合同的相关条款、性质、目的以及诚信原则等予以解释。

第四章 合同的履行

第五百零九条 当事人应当按照约定全面履行自己的义务。
当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。
当事人在履行合同过程中，应当避免浪费资源、污染环境和破坏生态。

第五百一十条 合同生效后，当事人就质量、价款或者报酬、履行地点等内容没有约定或者约定不明确的，可以协议补充；不能达成补充协议的，按照合同相关条款或者交易习惯确定。

第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。

第五百七十八条 当事人一方明确表示或者以自己的行为表明不履行合同义务的，对方可以在履行期限届满前请求其承担违约责任。

第五百七十九条 当事人一方未支付价款、报酬、租金、利息，或者不履行其他金钱义务的，对方可以请求其支付。

第五百八十条 当事一方不履行非金钱债务或者履行非金钱债务不符合约定的，对方可以请求履行，但是有下列情形之一的除外：
（一）法律上或者事实上不能履行；
（二）债务的标的不适于强制或者履行费用过高；
（三）债权人在合理期限内未请求履行。

第五百八十三条 当事人一方违约造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；但是，不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。

第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。
约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当降低。

第七章 合同的权利义务终止
第五百六十三条 有下列情形之一的，当事人可以解除合同：
（一）因不可抗力致使不能实现合同目的；
（二）在履行期限届满前，当事人一方明确表示或者以自己的行为表明不履行主要债务；
（三）当事人一方迟延履行主要债务，经催告后在合理期限内仍未履行；
（四）当事人一方迟延履行债务或者其他违约行为致使不能实现合同目的；
（五）法律规定的其他情形。
""",
            "metadata": {
                "doc_type": "law",
                "category": "民法-合同编",
                "source_code": "民法典_合同编",
                "effective_date": "2021-01-01",
            },
        },
        {
            "title": "中华人民共和国证券法（2019修订 摘录）",
            "content": """中华人民共和国证券法

第一章 总则

第一条 为了规范证券发行和交易行为，保护投资者的合法权益，维护社会经济秩序和社会公共利益，促进社会主义市场经济的发展，制定本法。

第二条 在中国境内，股票、公司债券、存托凭证和国务院依法认定的其他证券的发行和交易，适用本法；本法未规定的，适用《中华人民共和国公司法》和其他法律、行政法规的规定。

政府债券、证券投资基金份额的上市交易，适用本法；其他法律、行政法规另有规定的，适用其规定。

资产支持证券、资产管理产品发行、交易的管理办法，由国务院依照本法的原则规定。

第五十三条 证券交易内幕信息的知情人和非法获取内幕信息的人，在内幕信息公开前，不得买卖该公司的证券，或者泄露该信息，或者建议他人买卖该证券。
持有或者通过协议、其他安排与他人共同持有公司百分之五以上股份的自然人、法人、非法人组织收购上市公司的股份，本法另有规定的，适用其规定。

在内幕信息敏感期内，内幕信息知情人和非法获取内幕信息的人从事与该内幕信息有关的证券交易活动，所得收益归该公司所有。

第五十四条 证券交易活动中，涉及公司的经营、财务或者对该该公司证券的市场价格有重大影响的尚未公开的信息，为内幕信息。

第五十五条 下列人员为证券交易内幕信息的知情人：
（一）发行人及其董事、监事、高级管理人员；
（二）持有公司百分之五以上股份的股东及其董事、监事、高级管理人员，公司的实际控制人及其董事、监事、高级管理人员；
（三）发行人控股或者实际控制的公司及其董事、监事、高级管理人员；
（四）由于所任公司职务或者因与公司业务往来可以获取公司有关内幕信息的人员；
（五）上市公司收购人或者重大资产交易方及其控股股东、实际控制人、董事、监事、高级管理人员；
（六）因职务、工作可以获取内幕信息的证券监督管理机构工作人员；
（七）因法定职责对证券的发行、交易进行管理的其他人员；
（八）国务院证券监督管理机构规定的其他人。

第六十三条 发行人、上市公司或者其他信息披露义务人披露的信息，必须真实、准确、完整，不得有虚假记载、误导性陈述或者重大遗漏。

第六十九条 信息披露义务人披露的信息有虚假记载、误导性陈述或者重大遗漏，致使投资者在证券交易中遭受损失的，信息披露义务人应当承担赔偿责任；
发行人的控股股东、实际控制人、董事、监事、高级管理人员和其他直接责任人员以及保荐人、承销的证券公司及其直接责任人员，应当与发行人承担连带赔偿责任，但是能够证明自己没有过错的除外。

第一百九十三条 违反本法规定，操纵证券市场的，责令依法处理其非法持有的证券，没收违法所得，并处以违法所得一倍以下的罚款；没有违法所得或者违法所得不足三十万元的，处以三十万元以上三百万元以下的罚款。单位操纵证券市场的，还应当对其直接负责的主管人员和其他直接责任人员给予警告，并处以十万元以上六十万元以下的罚款。
""",
            "metadata": {
                "doc_type": "law",
                "category": "经济法-证券法",
                "source_code": "证券法_2019修订",
                "effective_date": "2020-03-01",
            },
        },
        {
            "title": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）（摘录）",
            "content": """最高人民法院关于审理劳动争议案件适用法律问题的解释（一）

为正确审理劳动争议案件，根据《中华人民共和国民法典》《中华人民共和国劳动法》《中华人民共和国劳动合同法》《中华人民共和国劳动争议调解仲裁法》《中华人民共和国民事诉讼法》等相关法律规定，结合审判实践，制定本解释。

第一条 劳动者与用人单位之间发生的下列纠纷，属于劳动争议，当事人不服劳动争议仲裁机构作出的裁决，依法提起诉讼的，人民法院应予受理：
（一）劳动者与用人单位在履行劳动合同过程中发生的纠纷；
（二）劳动者与用人单位之间没有订立书面劳动合同，但已形成劳动关系后发生的纠纷；
（三）劳动者与用人单位因劳动关系是否已经解除或者终止，以及应否支付解除或终止劳动关系经济补偿金发生的纠纷；
（四）劳动者与用人单位解除或者终止劳动关系后，请求用人单位返还劳动者收取的劳动合同定金、保证金、抵押金、抵押物发生的纠纷，或者办理劳动者的人事档案、社会保险关系等移转手续发生的纠纷；
（五）劳动者以用人单位未为其办理社会保险手续，且社会保险经办机构不能补办导致其无法享受社会保险待遇为由，要求用人单位赔偿损失发生的纠纷；
（六）企业改制引发职工分流安置等因自主裁员、变更劳动关系发生的纠纷；
（七）劳动者因为工伤、职业病，请求用人单位依法承担工伤保险待遇发生的纠纷；
（八）劳动者请求用人单位支付竞业限制经济补偿发生的纠纷；
（九）法律法规规定的其他劳动争议。

第二十三条 用人单位与劳动者约定了竞业限制条款，但未约定解除或者终止劳动合同后的竞业限制经济补偿的，劳动者履行了竞业限制义务的，可以要求用人单位按照劳动者在劳动合同解除或者终止前十二个月平均工资的30%按月支付经济补偿。

第二十四条 当事人在劳动合同或者保密协议中约定了竞业限制，同时约定了劳动合同解除或者终止后给予劳动者竞业限制经济补偿的，用人单位明确表示不支付经济补偿的，视为免除劳动者的竞业限制义务。

第三十九条 用人单位违反法律规定解除或者终止劳动合同的，应当依照劳动合同法第八十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。
""",
            "metadata": {
                "doc_type": "judicial",
                "category": "劳动法",
                "source_code": "最高法劳动争议解释一",
                "effective_date": "2021-01-01",
            },
        },
        {
            "title": "中华人民共和国反垄断法（2022修正 摘录）",
            "content": """中华人民共和国反垄断法

第一章 总则

第一条 为了预防和制止垄断行为，保护市场公平竞争，鼓励创新，提高经济运行效率，维护消费者利益和社会公共利益，促进社会主义市场经济健康发展，制定本法。

第二条 中华人民共和国境内经济活动中的垄断行为，适用本法；中华人民共和国境外的垄断行为，对境内市场竞争产生排除、限制影响的，适用本法。

第三条 本法规定的垄断行为包括：
（一）经营者达成垄断协议；
（二）经营者滥用市场支配地位；
（三）具有或者可能具有排除、限制竞争效果的经营者集中。

第十七条 禁止具有竞争关系的经营者达成下列垄断协议：
（一）固定或者变更商品价格；
（二）限制商品的生产数量或者销售数量；
（三）分割销售市场或者原材料采购市场；
（四）限制购买新技术、新设备或者限制开发新技术、新产品；
（五）联合抵制交易；
（六）国务院反垄断执法机构认定的其他垄断协议。

第十八条 禁止经营者与交易相对人达成下列垄断协议：
（一）固定向第三人转售商品的价格；
（二）限定向第三人转售商品的最低价格；
（三）国务院反垄断执法机构认定 的其他垄断协议。

第二十二条 具有市场支配地位的经营者不得滥用其市场支配地位，排除、限制竞争。禁止下列滥用市场支配地位的行为：
（一）以不公平的高价销售商品或者以不公平的低价购买商品；
（二）没有正当理由，以低于成本的价格销售商品；
（三）没有正当理由，拒绝与交易相对人进行交易；
（四）没有正当理由，限定交易相对人只能与其进行交易或者只能与其指定的经营者进行交易；
（五）没有正当理由搭售商品，或者在交易时附加其他不合理的交易条件；
（六）没有正当理由，对条件相同的交易相对人在交易价格等交易条件上实行差别待遇；
（七）国务院反垄断执法机构认定的其他滥用市场支配地位的行为。

第二十三条 有下列情形之一的，可以推定经营者具有市场支配地位：
（一）一个经营者在相关市场的市场份额达到二分之一的；
（二）两个经营者在相关市场的市场份额合计达到三分之二的；
（三个经营者者在相关市场的市场份额合计达到四分之三的。
有前款第二项、第三项规定的情形，其中有的经营者市场份额不足十分之一的，不应当推定该经营者具有市场支配地位。
""",
            "metadata": {
                "doc_type": "law",
                "category": "经济法-反垄断",
                "source_code": "反垄断法_2022修正",
                "effective_date": "2022-08-01",
            },
        },
    ]

    try:
        for law in demo_laws:
            await rag.ingest_text(law["content"], law["metadata"])
            logger.info(f"Seeded: {law['metadata'].get('source_code', 'unknown')}")

        return APIResponse(
            code=0,
            message=f"成功写入 {len(demo_laws)} 条示范性经济类法规（已通过 Pipeline 清洗解析）",
            data={"count": len(demo_laws), "laws": [l['metadata']['source_code'] for l in demo_laws]},
        )
    except Exception as e:
        logger.error(f"Seed error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 新增：Pipeline 清洗预览接口 ====================

@router.post("/preview-cleaning")
async def preview_cleaning(
    text: str = Form(..., description="待预览的原始文本"),
    filename: str = Form("sample.txt", description="模拟文件名"),
    request: Request = None,
):
    """
    文本清洗预览接口
    
    展示 Pipeline 对输入文本的处理效果（不入库，仅返回预览结果），用于：
      - 调试清洗效果
      - 对比清洗前后差异
      - 查看法条解析结果
    """
    try:
        ext = os.path.splitext(filename)[1].lower() or ".txt"

        # Step 1: 清洗
        cleaned_doc = LegalTextCleaner.clean(text, ext)

        # Step 2: 解析分块
        chunks = LegalTextParser.parse(cleaned_doc.cleaned_text, filename)

        # 构建预览报告
        chunk_preview = []
        for i, chunk in enumerate(chunks[:15]):  # 最多展示前15个分块
            chunk_preview.append({
                "index": i + 1,
                "article_number": chunk.article_number,
                "chapter": chunk.chapter or "-",
                "section": chunk.section or "-",
                "content_preview": chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
                "char_length": len(chunk.content),
            })

        return {
            "preview": True,
            "input_info": {
                "raw_length": len(text),
                "filename": filename,
            },
            "cleaning_stats": cleaned_doc.metadata,
            "parsed_chunks_total": len(chunks),
            "chunks_preview": chunk_preview,
            "detected_law_name": chunks[0].law_name if chunks else "未检测到",
            "detected_doc_type": chunks[0].doc_type if chunks else "unknown",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预览失败: {str(e)}")


# ==================== 新增：Pipeline 能力说明 ====================

@router.get("/pipeline-info")
async def pipeline_info(request: Request = None):
    """返回当前 Pipeline 的能力说明（含 Embedding 模型信息）"""
    # 获取当前 Embedding 信息
    embedding_info = {}
    if request:
        rag = request.app.state.rag_service
        try:
            embedding_info = rag.embedding_service.get_info() if rag.embedding_service else {}
        except Exception:
            pass
    
    return {
        "pipeline_version": "2.1",
        "embedding_model": embedding_info,  # ⭐ 新增：当前 Embedding 模型信息
        "stages": [
            {"stage": 1, "name": "文本提取", "formats": ["TXT", "MD", "PDF", "DOCX"], "engine": "TextExtractor"},
            {"stage": 2, "name": "文本清洗", "capabilities": ["PDF断行修复", "页面噪声去除", "标点规范化", "空白清理", "条文编号规范化"], "engine": "LegalTextCleaner"},
            {"stage": 3, "name": "结构解析", "capabilities": ["法条级分割(第X条模式)", "章节层级识别", "法律名称推断", "判例/合同回退分段"], "engine": "LegalTextParser"},
            {"stage": 4, "name": "元数据注入", "fields": ["article_number", "chapter", "section", "law_name", "doc_type", "char_length"]},
            {"stage": 5, "name": "向量化入库", "engine": f"{embedding_info.get('provider', 'Jina v2')} + Qdrant"},
        ],
        "supported_file_types": [".txt", ".md", ".pdf", ".docx"],
        "chunk_strategy": "article_first(法条优先) → paragraph_fallback(段落兜底)",
        "max_chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }


# ==================== 新增：Embedding 模型状态 ====================

@router.get("/embedding-status")
async def embedding_status(request: Request):
    """
    返回当前 Embedding 模型的详细状态
    
    用于前端展示当前使用的向量模型、维度、健康状态等信息，
    让用户明确知道检索质量依赖的是哪个模型。
    """
    rag = request.app.state.rag_service
    
    if not rag.embedding_service or not rag.embedding_service._initialized:
        return {
            "status": "not_initialized",
            "message": "Embedding 服务尚未初始化",
        }
    
    info = rag.embedding_service.get_info()
    healthy = await rag.embedding_service.health_check()
    
    return {
        "status": "ready" if healthy else "unhealthy",
        **info,
        "health_check": healthy,
        "note": "⚠️ Embedding 是专用向量编码模型，不是 LLM。FastEmbed 原生支持 jina-embeddings-v2-base-zh / bge-small-zh-v1.5。",
        "recommended_models": {
            "fastembed_zh_primary": "jinaai/jina-embeddings-v2-base-zh (1024维, 8K上下文, ★FastEmbed原生推荐)",
            "fastembed_zh_light": "BAAI/bge-small-zh-v1.5 (512维, ~90MB, FastEmbed原生, 最轻)",
            "st_bge_large": "BAAI/bge-large-zh-v1.5 (1024维, 需sentence-transformers后端)",
            "st_bge_m3": "BAAI/bge-m3 (1024维, 8K上下文, 效果最强, 需sentence-transformers)",
            "cloud_zhipu": "embedding-3 via 智谱AI (1024维)",
            "cloud_openai": "text-embedding-3-small via OpenAI (1536维)",
        },
    }
