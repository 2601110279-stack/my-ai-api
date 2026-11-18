# 📋 main.py
import base64
import binascii
import json
import os
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_B64 = 4_000_000  # Vercel 요청 본문 4.5MB 제한 대비

# ★ ① 앱 이름 · 아이콘 · 소개 (돈/가치 관련 주제이므로 '참고용' 표기 필수)
APP_TITLE = "방구석 진품명품 · 덕후 감정사"
APP_ICON = "🧐"
APP_DESCRIPTION = "피규어, 굿즈, 전자기기 등 소장품 사진을 찍으면 진품명품 전문가처럼 연대와 가상의 소장 가치를 감정해 줍니다. (본 감정가는 재미로 보는 참고용입니다.)"

# ★ ② AI에게 내리는 지시문 (지어내지 않기 규칙 포함)
PROMPT = """너는 TV 프로그램 '진품명품'에 나오는 깐깐하고 유머러스한 덕후 전문 감정사야.
사용자가 보낸 사진 속 물건의 외관, 재질, 희소성, 관리 상태를 꼼꼼히 관찰해서 진지하면서도 재치 있는 전문가 말투('~하옵니다', '~로 보입니다' 등)로 감정해 줘.
사진에서 확인할 수 없는 부분은 절대 지어내지 말고 '확인 불가'라고 적어.
감정가는 재미를 위한 가상의 금액으로 책정하되 반드시 뒤에 '(참고용)'을 붙여 줘."""

# ★ ③ 결과 표에 나올 칸 이름과 설명 (3~5개)
FIELDS = {
    "감정 품목": "물건의 정확한 명칭과 추정 연대",
    "보존 상태": "외관 스크래치, 사용감, 관리 상태에 대한 깐깐한 평가",
    "감정가": "재미로 매긴 가상의 소장 가치 (예: 1,500만 원 (참고용))",
    "감정사 총평": "진품명품 전문가 말투의 유머러스한 한 줄 평",
}

SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string", "description": desc}
        for key, desc in FIELDS.items()
    },
    "required": list(FIELDS.keys()),
}

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)


class AnalyzeRequest(BaseModel):
    image_b64: str = Field(min_length=1, description="사진(JPEG · PNG)을 base64로 바꾼 글자")
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"


class AnalyzeResponse(BaseModel):
    result: dict[str, str]
    tokens: int


def check_access(x_access_code: str | None):
    expected = os.environ.get("APP_ACCESS_CODE", "")
    if not expected or x_access_code != expected:
        raise HTTPException(status_code=401, detail="접근 코드가 틀렸습니다")


def check_b64(text: str):
    if len(text) > MAX_B64:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 — 사진을 줄여 보내세요")
    try:
        base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="base64 글자가 아닙니다")


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="서버에 GEMINI_API_KEY가 없습니다")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )


@app.get("/")
def home():
    return {
        "name": APP_TITLE,
        "icon": APP_ICON,
        "description": APP_DESCRIPTION,
        "fields": FIELDS,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, x_access_code: str | None = Header(default=None)):
    check_access(x_access_code)
    check_b64(request.image_b64)
    client = get_client()
    try:
        reply = client.interactions.create(
            model=MODEL_NAME,
            store=False,
            input=[
                {"type": "text", "text": PROMPT},
                {"type": "image", "data": request.image_b64, "mime_type": request.mime_type},
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SCHEMA,
            },
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Gemini 요청 실패: {error}")

    parsed = json.loads(reply.output_text)
    tokens = getattr(reply.usage, "total_tokens", 0)
    return AnalyzeResponse(result=parsed, tokens=tokens)
