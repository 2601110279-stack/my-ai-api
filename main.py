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
PROMPT = """
너는 피규어, 굿즈, 전자기기 등 다양한 중고 물품을
감정하는 전문가야.

사용자가 올린 사진을 분석하여 다음 항목을 작성해 줘.

1. 감정 품목
- 사진 속 물품의 종류와 특징을 설명해.
- 브랜드나 제품명이 확인되면 작성해.
- 정확한 제품명을 알 수 없으면 추정이라고 표시해.

2. 보존 상태
- 사진에서 확인되는 오염, 흠집, 파손 등을 평가해.
- 상태를 최상, 상, 중, 하 중 하나로 판단해.
- 사진에 보이지 않는 부분은 확인할 수 없다고 설명해.

3. 감정가
- 물품의 종류, 브랜드, 예상 연식, 외관 상태,
  희소성 등을 고려하여 예상 중고 가격을 추정해.
- 모든 물품에 똑같은 금액을 적용하지 마.
- 물품별 특성과 상태를 개별적으로 고려해.
- 감정가는 원화(KRW)로 작성해.
- 사진만으로 가격을 판단하기 어려우면
  '가격 추정 어려움'이라고 작성해.
- 실제 거래 시세를 조회하지 않았다면
  조회한 것처럼 설명하지 마.
  - 가격을 추정할 근거가 충분하다면
  단일 금액보다는 예상 가격 범위로 제시해.
- 물품의 종류와 상태에 따라 가격 범위를 다르게 판단해.
- 비슷한 가치를 가진 물품은 비슷한 가격이 나와도 괜찮아.
- 가격 차이를 만들기 위해 임의로 금액을 변경하지 마.
- 감정가의 근거가 부족하면 무리하게 숫자를 제시하지 마.

4. 감정사 총평
- 해당 감정가를 추정한 이유를 2~3문장으로 설명해.
- 수집 가치와 상태를 함께 고려해.

사진과 무관한 내용을 만들어 내지 말고,
불확실한 내용은 추정이라고 명확하게 표시해.
"""


# ★ ③ 결과 표에 나올 칸 이름과 설명 (3~5개)
FIELDS = {
    "감정 품목": "물품 종류, 브랜드, 제품 특징 및 예상 연대",
    "보존 상태": "외관 상태와 손상 여부 및 상태 등급",
    "감정가": "예상 중고 가격 범위(원화)",
    "감정사 총평": "가격 추정 근거와 수집 가치에 대한 설명",
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

    try:
        parsed = json.loads(reply.output_text)

        if not isinstance(parsed, dict):
            raise ValueError("JSON 객체가 아닙니다")

        if not all(
            isinstance(parsed.get(key), str)
            for key in FIELDS
        ):
            raise ValueError("필수 감정 항목이 누락되었습니다")

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Gemini 분석 결과를 처리하지 못했습니다"
        )

    usage = getattr(reply, "usage", None)
    tokens = getattr(usage, "total_tokens", 0) or 0

    return AnalyzeResponse(
        result=parsed,
        tokens=tokens
    )
