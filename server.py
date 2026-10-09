"""
风格转换系统 - FastAPI 后端服务
"""
import os
import sys
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="风格转换系统", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型（延迟加载）
pipeline = None


def get_pipeline():
    """获取风格转换管线（延迟加载）"""
    global pipeline
    if pipeline is None:
        from src.pipeline import StyleTransferPipeline
        
        base_model = "models/Qwen2.5-1.5B-Instruct"
        lora_weights = "outputs/lora_checkpoint_v2/final_lora_weights"
        
        if not os.path.exists(base_model):
            raise FileNotFoundError(f"基座模型不存在: {base_model}")
        if not os.path.exists(lora_weights):
            raise FileNotFoundError(f"LoRA 权重不存在: {lora_weights}")
        
        pipeline = StyleTransferPipeline(
            base_model_path=base_model,
            lora_weights_path=lora_weights,
            device="auto"
        )
    return pipeline


class TransferRequest(BaseModel):
    """转换请求"""
    text: str
    style: str
    max_tokens: int = 150
    temperature: float = 0.8


class TransferResponse(BaseModel):
    """转换响应"""
    original: str
    style: str
    result: str
    semantic_info: Optional[dict] = None


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse("static/index.html")


@app.post("/api/transfer", response_model=TransferResponse)
async def style_transfer(request: TransferRequest):
    """风格转换 API"""
    try:
        p = get_pipeline()
        result = p.transfer(
            text=request.text,
            style=request.style,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return TransferResponse(
            original=result["original_text"],
            style=result["target_style"],
            result=result["generated_text"],
            semantic_info=result.get("semantic_info")
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@app.get("/api/styles")
async def list_styles():
    """返回支持的风格列表"""
    return {
        "styles": [
            {"id": "正式", "name": "正式", "description": "书面化、正式的表达方式"},
            {"id": "口语化", "name": "口语化", "description": "日常对话、轻松自然的风格"},
            {"id": "简洁", "name": "简洁", "description": "精炼、去掉冗余修饰"},
            {"id": "丰富", "name": "丰富", "description": "扩展细节、增加修饰"},
        ]
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model_loaded": pipeline is not None}


# 挂载静态文件
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("风格转换系统 - Web 服务")
    print("=" * 60)
    print("启动中...")
    print("请在浏览器中访问: http://localhost:8000")
    print("API 文档: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
