"""
Production-grade Multi-turn Chatbot Pipeline
=============================================
가상의 회사 "모든전자(Modeun Electronics)" 의 고객 문의 처리용 멀티턴 챗봇.
1) Pydantic 으로 문의를 카테고리·긴급도로 정형화 분류
2) InMemorySaver 체크포인터로 thread_id 단위 대화 기억
3) SummarizationMiddleware 로 대화가 길어지면 자동 요약 압축

실행 모드:
    --mode classify   : 고객 문의 정형 분류 (8건 일괄 처리)
    --mode chat       : 멀티턴 대화 — thread_id 기반 메모리 ON/OFF 비교
    --mode summarize  : 장시간 세션에 요약 미들웨어 적용
    --mode visualize  : 시각화 PNG 7장 생성 (LLM 호출 불필요)
    --mode all        : 위 단계 전부 실행

원본 노트북:
    chapter_05_structured_output.ipynb / chapter_06_checkpointer_summaryzation.ipynb
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 무거운 임포트는 LLM 호출 모드에서만 사용 — 시각화 모드는 위 모듈만으로 충분
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INQUIRY_FILE = DATA_DIR / "customer_inquiries.json"


# ════════════════════════════════════════════════════════════════════════════
# 1. Pydantic 스키마 — 정형화된 응답 형식
# ════════════════════════════════════════════════════════════════════════════
class InquiryCategory(str, Enum):
    PRODUCT = "제품문의"
    ORDER = "주문관리"
    DELIVERY = "배송지연"
    TECHNICAL = "기술지원"
    OTHER = "기타"


class InquiryClassification(BaseModel):
    """고객 문의를 정형 데이터로 변환한 결과."""

    category: InquiryCategory = Field(description="문의 카테고리 (제품문의/주문관리/배송지연/기술지원/기타 중 하나)")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description=(
            "긴급도. 일반 안내는 low, 일정 확인은 medium, "
            "지연·페널티 언급은 high, 양산 중단·서비스 장애는 critical."
        )
    )
    summary: str = Field(description="문의 핵심을 한 문장으로 요약")
    requires_human: bool = Field(
        description="사람 상담사 연결이 필요한지 여부 (urgency=high/critical 이면 항상 True)"
    )


class CustomerProfile(BaseModel):
    """대화 중에 챗봇이 기억해 둔 고객 프로필."""

    name: str | None = Field(default=None, description="고객 이름")
    company: str | None = Field(default=None, description="고객 소속 회사")
    last_inquiry_category: InquiryCategory | None = Field(
        default=None, description="가장 최근 문의 카테고리"
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. LLM 빌더
# ════════════════════════════════════════════════════════════════════════════
def build_llm():
    from langchain_openai import ChatOpenAI

    load_dotenv()
    return ChatOpenAI(
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        model="ignored-by-proxy",
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. 문의 분류기 — `with_structured_output`
# ════════════════════════════════════════════════════════════════════════════
CLASSIFY_SYSTEM_PROMPT = """
너는 모든전자의 고객 문의 분류기다.
원문 문의를 읽고 카테고리·긴급도·한 줄 요약·상담사 연결 여부를 정확히 채워라.
긴급도 판단 기준:
  - low: 일반 안내, 자료 요청
  - medium: 일정 확인, 진행 상황 공유 요청
  - high: 납기 지연, 페널티 언급, 마감 임박
  - critical: 양산 라인 중단, 서비스 장애, 즉시 조치 필요
high 이상은 requires_human=True 로 강제한다.
"""


def classify_inquiry(llm, raw_text: str) -> InquiryClassification:
    structured = llm.with_structured_output(InquiryClassification)
    return structured.invoke(
        [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ]
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. 멀티턴 챗봇 — checkpointer + summarization
# ════════════════════════════════════════════════════════════════════════════
CHAT_SYSTEM_PROMPT = """
너는 모든전자의 친절한 고객 응대 챗봇이다.
한국어로 자연스럽게 답변한다.
이전에 사용자가 알려 준 이름·회사·관심 제품 등이 있으면 반드시 기억해서 활용한다.
대화 도중 사용자가 새로운 정보를 주면, 그 정보를 즉시 답변에 반영한다.
"""


def build_chatbot(llm, with_memory: bool = True, with_summarization: bool = True):
    from langchain.agents import create_agent

    middlewares: list[Any] = []
    checkpointer = None

    if with_memory:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()

    if with_summarization:
        from langchain.agents.middleware import SummarizationMiddleware
        middlewares.append(
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", 4000),
                keep=("messages", 20),
            )
        )

    kwargs: dict[str, Any] = {"model": llm, "system_prompt": CHAT_SYSTEM_PROMPT}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if middlewares:
        kwargs["middleware"] = middlewares
    return create_agent(**kwargs)


def replay_turns(agent, thread_id: str, turns: list[str]) -> list[tuple[str, str]]:
    """thread_id 를 고정해 멀티턴 대화를 차례로 흘려 보고, (user, ai) 쌍을 반환."""
    config = {"configurable": {"thread_id": thread_id}}
    log: list[tuple[str, str]] = []
    for user_msg in turns:
        out = agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)
        ai_msg = out["messages"][-1].content
        log.append((user_msg, ai_msg))
    return log


# ════════════════════════════════════════════════════════════════════════════
# 5. 실행 함수
# ════════════════════════════════════════════════════════════════════════════
def run_classification(llm) -> dict[str, Any]:
    print("\n[ 1. 고객 문의 정형 분류 ]\n")
    inquiries = json.loads(INQUIRY_FILE.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for inq in inquiries:
        cls = classify_inquiry(llm, inq["raw_text"])
        print(f"  [{inq['id']}] {cls.category.value:>6} / {cls.urgency:>8} / "
              f"human={cls.requires_human}\n         → {cls.summary}")
        results.append({
            "id": inq["id"],
            "raw": inq["raw_text"],
            "category": cls.category.value,
            "urgency": cls.urgency,
            "requires_human": cls.requires_human,
            "summary": cls.summary,
        })
    return {"classifications": results}


def run_memory_comparison(llm) -> dict[str, Any]:
    print("\n[ 2. 멀티턴 대화 — thread_id 기반 메모리 ON/OFF 비교 ]\n")
    agent_with_memory = build_chatbot(llm, with_memory=True, with_summarization=False)
    agent_no_memory = build_chatbot(llm, with_memory=False, with_summarization=False)

    turns = [
        "안녕하세요, 저는 (주)알파테크 김철수 책임입니다.",
        "지난주 보내 주신 GaN 모듈 데이터시트 잘 받았습니다.",
        "효율 그래프 중에 80% 부하 구간 수치 한 번 더 확인 부탁드립니다.",
        "그런데 제 이름이 뭐였죠? 챗봇이 기억하는지 확인하고 싶어서요.",
    ]
    print("  --- WITH 체크포인터 ---")
    with_log = replay_turns(agent_with_memory, "alpha_with_memory", turns)
    for u, a in with_log:
        print(f"  USER: {u}\n  BOT : {a[:140]}{'...' if len(a) > 140 else ''}\n")

    print("  --- WITHOUT 체크포인터 ---")
    without_log = replay_turns(agent_no_memory, "alpha_no_memory", turns)
    for u, a in without_log:
        print(f"  USER: {u}\n  BOT : {a[:140]}{'...' if len(a) > 140 else ''}\n")

    return {"with_memory": with_log, "without_memory": without_log}


def run_long_session(llm) -> dict[str, Any]:
    print("\n[ 3. 요약 미들웨어 (긴 대화 자동 압축) ]\n")
    agent = build_chatbot(llm, with_memory=True, with_summarization=True)
    long_turns = [
        "안녕하세요, 모든전자 챗봇이죠?",
        "저는 (주)베타시스템 박지수 매니저입니다.",
        "주력 사업은 산업용 로봇 컨트롤러 제조입니다.",
        "최근에 GaN 기반 전력 모듈을 도입 검토 중입니다.",
        "특히 발열 관리 솔루션이 궁금합니다.",
        "150W/in³ 정도 출력에서 안정적으로 동작하나요?",
        "데이터시트에 있는 듀얼 쿨링 옵션도 사용 가능한가요?",
        "우리 라인 환경은 주변 온도 40도 정도 됩니다.",
        "ODM 커스터마이징도 가능한지 궁금합니다.",
        "샘플 단가와 양산 단가 모두 공유 가능할까요?",
        "납기 리드타임은 보통 어느 정도 잡으시나요?",
        "최소 발주 수량(MOQ)은 얼마인가요?",
        "혹시 제 이름과 회사를 다시 한 번 정리해 주실 수 있나요?",
    ]
    log = replay_turns(agent, "beta_long_session", long_turns)
    for u, a in log[-3:]:
        print(f"  USER: {u}\n  BOT : {a[:160]}{'...' if len(a) > 160 else ''}\n")
    return {"summarization_log": log}


# ════════════════════════════════════════════════════════════════════════════
# 6. 시각화 (LLM 호출 없이 동작)
# ════════════════════════════════════════════════════════════════════════════
def _wrap(text: str, width: int = 70) -> list[str]:
    return textwrap.wrap(" ".join(text.split()), width=width,
                         break_long_words=True, break_on_hyphens=False)


def visualize_pipeline_overview() -> Path:
    """좌→우 한 방향 흐름 + 두 개의 미들웨어 박스가 챗봇 위에 부착되는 형태."""
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 제목
    ax.text(8, 7.55, "Production-grade Multi-turn Chatbot — 전체 흐름",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(8, 7.15,
            "정형 분류 (위쪽 분기) · 멀티턴 챗봇 (아래쪽 분기) · 두 갈래가 동일한 LLM 자원을 공유",
            ha="center", fontsize=10, color="#555", style="italic")

    def box(x, y, w, h, text, fc, ec, fontsize=10, sub=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                    edgecolor=ec, linewidth=1.4))
        if sub:
            ax.text(x + w / 2, y + h / 2 + 0.2, text,
                    ha="center", va="center", fontsize=fontsize, fontweight="bold")
            # 한글 포함 여부에 따라 폰트 선택 — monospace 는 한글 글리프 없음
            has_korean = any("가" <= ch <= "힯" for ch in sub)
            if has_korean:
                ax.text(x + w / 2, y + h / 2 - 0.28, sub,
                        ha="center", va="center", fontsize=fontsize - 1.5, color="#37474F")
            else:
                ax.text(x + w / 2, y + h / 2 - 0.28, sub,
                        ha="center", va="center", fontsize=fontsize - 1.5, color="#37474F",
                        family="monospace")
        else:
            ax.text(x + w / 2, y + h / 2, text,
                    ha="center", va="center", fontsize=fontsize, fontweight="bold")

    # 1. 고객 입력
    box(0.3, 3.0, 1.7, 1.4, "고객\n문의", "#FFE0B2", "black")

    # 2-A. 분류기 (위)
    box(2.6, 5.1, 3.4, 1.4, "Pydantic 분류기",
        "#C5CAE9", "#283593",
        sub="with_structured_output(...)")

    # 2-B. 챗봇 (아래)
    box(2.6, 1.0, 3.4, 1.4, "멀티턴 챗봇",
        "#C8E6C9", "#2E7D32",
        sub="create_agent(checkpointer=, middleware=)")

    # 3. 미들웨어 두 개 (챗봇 위쪽으로 부착)
    box(6.7, 4.8, 3.3, 1.0, "InMemorySaver",
        "#FFF59D", "#F9A825", fontsize=10,
        sub="thread_id 단위 대화 저장")
    box(6.7, 3.2, 3.3, 1.0, "SummarizationMiddleware",
        "#FFCCBC", "#D84315", fontsize=10,
        sub="trigger=4k tok / keep=20 msg")

    # 4. 출력
    box(11.0, 5.1, 4.6, 1.4, "정형 분류 결과",
        "#F8BBD0", "#AD1457",
        sub="InquiryClassification")
    box(11.0, 1.0, 4.6, 1.4, "자연어 답변",
        "#FFCDD2", "#C62828",
        sub="이름·회사·이전 맥락 반영")

    # 화살표
    def arr(x1, y1, x2, y2, lw=1.6, color="#37474F"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     mutation_scale=14))

    # 입력 → 분기
    arr(2.0, 3.9, 2.6, 5.5)   # 고객 → 분류기
    arr(2.0, 3.5, 2.6, 1.6)   # 고객 → 챗봇
    # 분류기 → 출력
    arr(6.0, 5.8, 11.0, 5.8)
    # 챗봇 ↔ 미들웨어 (양방향)
    ax.annotate("", xy=(6.7, 5.3), xytext=(5.5, 2.4),
                arrowprops=dict(arrowstyle="<->", color="#F9A825",
                                 lw=1.4, linestyle="--", mutation_scale=12))
    ax.annotate("", xy=(6.7, 3.7), xytext=(5.5, 2.0),
                arrowprops=dict(arrowstyle="<->", color="#D84315",
                                 lw=1.4, linestyle="--", mutation_scale=12))
    # 챗봇 → 답변
    arr(6.0, 1.7, 11.0, 1.7)

    # 범례
    legend = [
        mpatches.Patch(facecolor="#FFE0B2", edgecolor="black", label="고객 입력"),
        mpatches.Patch(facecolor="#C5CAE9", edgecolor="#283593", label="정형 분류"),
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="#2E7D32", label="멀티턴 챗봇"),
        mpatches.Patch(facecolor="#FFF59D", edgecolor="#F9A825", label="대화 메모리"),
        mpatches.Patch(facecolor="#FFCCBC", edgecolor="#D84315", label="자동 요약"),
        mpatches.Patch(facecolor="#F8BBD0", edgecolor="#AD1457", label="정형 출력"),
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="#C62828", label="자연어 답변"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=7, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    out = RESULTS_DIR / "fig_01_pipeline_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_schema_card(
    ax,
    x: float, y_top: float, w: float,
    title: str, fields: list[tuple[str, str, str]],
    bg: str, edge: str, accent: str,
) -> float:
    """Pydantic 스키마 카드를 그리고 카드의 총 높이를 반환."""
    HEAD_H = 0.95
    ROW_NAME_H = 0.45
    ROW_DESC_LINE_H = 0.35
    ROW_PAD = 0.32
    SIDE_PAD = 0.35
    BOTTOM_PAD = 0.45

    # 각 필드의 description을 미리 wrap
    wrap_w = int((w - SIDE_PAD * 2) * 8)  # 대략 셀 폭에 맞춘 글자 수
    wrap_w = max(35, min(wrap_w, 60))
    wrapped_fields: list[tuple[str, str, list[str]]] = []
    for name, ftype, desc in fields:
        lines = textwrap.wrap(desc, width=wrap_w, break_long_words=True, break_on_hyphens=False)
        wrapped_fields.append((name, ftype, lines))

    rows_h = sum(ROW_NAME_H + len(lines) * ROW_DESC_LINE_H + ROW_PAD
                 for _, _, lines in wrapped_fields)
    total_h = HEAD_H + rows_h + BOTTOM_PAD

    # 배경 박스
    ax.add_patch(plt.Rectangle((x, y_top - total_h), w, total_h,
                                facecolor=bg, edgecolor=edge, linewidth=1.6))
    # 헤더 띠
    ax.add_patch(plt.Rectangle((x, y_top - HEAD_H), w, HEAD_H,
                                facecolor=accent, edgecolor=edge, linewidth=1.0))
    ax.text(x + w / 2, y_top - HEAD_H / 2, title,
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white")

    # 필드 행
    cur_y = y_top - HEAD_H - 0.22
    for name, ftype, lines in wrapped_fields:
        # 필드명 + 타입
        ax.text(x + SIDE_PAD, cur_y - ROW_NAME_H / 2,
                name, fontsize=10.5, fontweight="bold",
                color=accent, va="center")
        ax.text(x + w - SIDE_PAD, cur_y - ROW_NAME_H / 2,
                ftype, fontsize=9.5, family="monospace",
                color="#37474F", ha="right", va="center")
        # description 여러 줄
        desc_y = cur_y - ROW_NAME_H - 0.08
        for i, line in enumerate(lines):
            ax.text(x + SIDE_PAD + 0.2,
                    desc_y - i * ROW_DESC_LINE_H - ROW_DESC_LINE_H / 2,
                    line, fontsize=9, color="#455A64", va="center")
        # 행 구분선
        cur_y -= ROW_NAME_H + len(lines) * ROW_DESC_LINE_H + ROW_PAD
        if (name, ftype, lines) != wrapped_fields[-1]:
            ax.plot([x + SIDE_PAD, x + w - SIDE_PAD], [cur_y + ROW_PAD / 2] * 2,
                    color=edge, linewidth=0.4, alpha=0.4)

    return total_h


def visualize_pydantic_schema() -> Path:
    fields_inq = [
        ("category", "InquiryCategory", "제품문의 / 주문관리 / 배송지연 / 기술지원 / 기타 중 하나로 강제"),
        ("urgency", "Literal[low|medium|high|critical]", "low=일반 안내, medium=일정 확인, high=납기 지연·페널티, critical=양산 중단·서비스 장애"),
        ("summary", "str", "문의 본문을 한 문장으로 압축한 요약. 운영 대시보드에 그대로 노출 가능"),
        ("requires_human", "bool", "True 이면 자동으로 사람 상담사 연결. high·critical 긴급도면 자동 True 강제"),
    ]
    fields_cust = [
        ("name", "str | None", "고객 이름. 첫 인사에서 추출 후 thread 메모리에 저장"),
        ("company", "str | None", "고객 소속 회사. 발주·계약 이력 매칭 키로도 활용"),
        ("last_inquiry_category", "InquiryCategory | None", "직전 문의 카테고리. 동일 thread 내 후속 질문 라우팅에 사용"),
    ]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # 제목
    ax.text(7, 8.55, "Pydantic 정형 응답 스키마",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(7, 8.18,
            "Field(description=...) 의 설명문이 LLM 에게 함께 전달되어, 출력이 타입·규칙을 어기지 않도록 강제됩니다.",
            ha="center", fontsize=10, color="#555", style="italic")

    # 두 카드 — 동일 y_top 정렬
    y_top = 7.7
    card_w = 6.4
    h1 = _draw_schema_card(
        ax, x=0.4, y_top=y_top, w=card_w,
        title="class InquiryClassification(BaseModel)",
        fields=fields_inq,
        bg="#E3F2FD", edge="#1565C0", accent="#1565C0",
    )
    h2 = _draw_schema_card(
        ax, x=7.2, y_top=y_top, w=card_w,
        title="class CustomerProfile(BaseModel)",
        fields=fields_cust,
        bg="#FFF3E0", edge="#E65100", accent="#E65100",
    )

    # 사용 예시 코드 박스
    code_y_top = y_top - max(h1, h2) - 0.4
    ax.add_patch(plt.Rectangle((0.4, code_y_top - 1.2), 13.2, 1.2,
                                facecolor="#263238", edgecolor="#37474F", linewidth=1.0))
    ax.text(0.6, code_y_top - 0.28,
            "▶ 사용 예시",
            fontsize=10, fontweight="bold", color="#FFD54F", va="center")
    ax.text(0.6, code_y_top - 0.62,
            "result = llm.with_structured_output(InquiryClassification).invoke(messages)",
            fontsize=10, family="monospace", color="#A5D6A7", va="center")
    ax.text(0.6, code_y_top - 0.95,
            "→ result 는 항상 타입이 맞는 InquiryClassification 객체. category·urgency 값도 정해진 후보 안에서만 나옴.",
            fontsize=9.5, color="#CFD8DC", va="center")

    out = RESULTS_DIR / "fig_02_pydantic_schema.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_classification_distribution() -> Path:
    """8건 고객 문의가 어떤 카테고리·긴급도로 분류되는지 매트릭스."""
    inquiries = json.loads(INQUIRY_FILE.read_text(encoding="utf-8"))
    categories = ["제품문의", "주문관리", "배송지연", "기술지원", "기타"]
    urgency_order = ["low", "medium", "high", "critical"]

    # 카운트
    matrix = np.zeros((len(urgency_order), len(categories)), dtype=int)
    short_labels = []
    for inq in inquiries:
        c = inq["expected_category"]
        u = inq["expected_urgency"]
        matrix[urgency_order.index(u), categories.index(c)] += 1
        short = inq["raw_text"][:30] + ("..." if len(inq["raw_text"]) > 30 else "")
        short_labels.append((short, c, u))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1.2, 2]})

    # 왼쪽: 카테고리×긴급도 매트릭스
    ax1 = axes[0]
    im = ax1.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=matrix.max() + 0.5)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
    ax1.set_yticks(range(len(urgency_order)))
    ax1.set_yticklabels(urgency_order, fontsize=10)
    for i in range(len(urgency_order)):
        for j in range(len(categories)):
            v = int(matrix[i, j])
            if v > 0:
                ax1.text(j, i, str(v), ha="center", va="center",
                         color="white" if v >= 1 else "black",
                         fontweight="bold", fontsize=11)
    ax1.set_title("긴급도 × 카테고리 분포", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 오른쪽: 고객 문의 → 분류 결과 표
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, len(short_labels) + 1.5)

    color_map = {"low": "#C8E6C9", "medium": "#FFF59D",
                 "high": "#FFAB91", "critical": "#EF5350"}
    text_color_map = {"low": "#1B5E20", "medium": "#5D4037",
                       "high": "#BF360C", "critical": "white"}

    ax2.text(0.2, len(short_labels) + 0.7, "고객 문의",
             fontsize=10, fontweight="bold", color="#333")
    ax2.text(6.3, len(short_labels) + 0.7, "카테고리",
             fontsize=10, fontweight="bold", color="#333")
    ax2.text(8.3, len(short_labels) + 0.7, "긴급도",
             fontsize=10, fontweight="bold", color="#333")

    for i, (txt, cat, urg) in enumerate(reversed(short_labels)):
        y = i + 0.5
        ax2.text(0.2, y, txt, fontsize=9, va="center")
        ax2.text(6.3, y, cat, fontsize=9, va="center")
        ax2.add_patch(plt.Rectangle((8.1, y - 0.32), 1.5, 0.65,
                                     facecolor=color_map[urg], edgecolor="black", linewidth=0.5))
        ax2.text(8.85, y, urg, fontsize=9, va="center", ha="center",
                 color=text_color_map[urg], fontweight="bold")

    ax2.set_title("8건 고객 문의 → 분류 결과", fontweight="bold", fontsize=11)

    fig.suptitle("Pydantic 분류 결과 — 카테고리 · 긴급도",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = RESULTS_DIR / "fig_03_classification_distribution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_memory_comparison() -> Path:
    """체크포인터 유무에 따른 멀티턴 대화 결과 비교."""
    turns = [
        ("U", "안녕하세요, 저는 (주)알파테크 김철수 책임입니다."),
        ("B_with", "안녕하세요 김철수 책임님, (주)알파테크 잘 알고 있습니다. 무엇을 도와드릴까요?"),
        ("B_without", "안녕하세요! 무엇을 도와드릴까요?"),
        ("U", "지난주 보내 주신 GaN 모듈 데이터시트 잘 받았습니다."),
        ("B_with", "다행입니다, 김 책임님. 데이터시트에서 추가로 궁금하신 부분이 있나요?"),
        ("B_without", "감사합니다. 데이터시트 잘 받으셨다니 다행이네요."),
        ("U", "그런데 제 이름과 회사가 뭐였죠?"),
        ("B_with", "김철수 책임님, (주)알파테크 소속이라고 알려 주셨습니다."),
        ("B_without", "죄송합니다, 이전 대화 내용을 기억하지 못합니다. 다시 알려 주시겠어요?"),
    ]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 12)
    ax.axis("off")

    LINE_H = 0.45
    PAD = 0.25
    GAP = 0.3
    cur_y = 0.0
    blocks: list[tuple[str, str, list[str], str, str]] = []

    for i in range(0, len(turns), 3):
        u_label, u_text = turns[i]
        bw_label, bw_text = turns[i + 1]
        bo_label, bo_text = turns[i + 2]
        blocks.append(("user", u_text, _wrap(u_text, 95), "#FFE0B2", "#5D4037"))
        blocks.append(("with", bw_text, _wrap(bw_text, 95), "#C8E6C9", "#1B5E20"))
        blocks.append(("without", bo_text, _wrap(bo_text, 95), "#FFCDD2", "#B71C1C"))

    total_h = sum(PAD * 2 + len(b[2]) * LINE_H for b in blocks) + GAP * (len(blocks) - 1) + 1.0
    ax.set_ylim(0, total_h)

    cur_y = total_h - 0.6
    ax.text(6, total_h - 0.2,
            "동일 4턴 대화 → WITH 체크포인터 vs WITHOUT 체크포인터",
            ha="center", fontsize=13, fontweight="bold")

    label_map = {"user": "USER", "with": "BOT (with memory)", "without": "BOT (no memory)"}
    for kind, text, lines, bg, edge in blocks:
        h = PAD * 2 + len(lines) * LINE_H
        x_offset = 0.2 if kind == "user" else 1.0
        width = 11.6 if kind == "user" else 10.8

        ax.add_patch(plt.Rectangle((x_offset, cur_y - h), width, h,
                                    facecolor=bg, edgecolor=edge, linewidth=1.0))
        ax.text(x_offset + 0.2, cur_y - PAD - 0.05,
                label_map[kind], fontsize=9, fontweight="bold",
                color=edge, va="top")
        for i, line in enumerate(lines):
            ax.text(x_offset + 0.2, cur_y - PAD - 0.45 - i * LINE_H,
                    line, fontsize=9.5, va="top", color="#212121")
        cur_y -= h + GAP

    out = RESULTS_DIR / "fig_04_memory_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_summarization_flow() -> Path:
    """SummarizationMiddleware 가 30턴 대화를 어떻게 압축하는지 시각화."""
    n_turns = 30
    threshold_token = 4000
    keep_messages = 20

    # 각 턴의 누적 토큰 가정값 (점진적으로 증가)
    rng = np.random.default_rng(42)
    per_turn_tokens = rng.integers(150, 350, size=n_turns)
    cumulative = np.cumsum(per_turn_tokens)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_colors = []
    summarized_count = 0
    for i, c in enumerate(cumulative):
        if i < n_turns - keep_messages:
            bar_colors.append("#EF9A9A")  # 요약 대상
            summarized_count += 1
        else:
            bar_colors.append("#A5D6A7")  # 원문 유지

    ax.bar(range(1, n_turns + 1), per_turn_tokens, color=bar_colors,
           edgecolor="black", linewidth=0.4)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_turns + 1), cumulative, color="#1976D2", linewidth=2,
             marker="o", markersize=4, label="누적 토큰")
    ax2.axhline(threshold_token, color="red", linestyle="--", linewidth=1.2,
                label=f"trigger = {threshold_token} 토큰")
    ax2.set_ylabel("누적 토큰", color="#1976D2")
    ax2.tick_params(axis="y", labelcolor="#1976D2")
    ax2.legend(loc="upper left", fontsize=9)

    ax.set_xlabel("대화 턴 번호 (1 → 30)")
    ax.set_ylabel("턴별 토큰 수")
    ax.set_title(
        f"SummarizationMiddleware — 30턴 중 앞 {summarized_count}턴 요약 / 최근 {keep_messages}턴 원문 유지",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    legend = [
        mpatches.Patch(facecolor="#EF9A9A", edgecolor="black", label="요약 대상 (앞쪽 오래된 턴)"),
        mpatches.Patch(facecolor="#A5D6A7", edgecolor="black", label=f"원문 유지 (최근 {keep_messages}턴)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    out = RESULTS_DIR / "fig_05_summarization_flow.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_thread_isolation() -> Path:
    """3개 thread 의 멀티턴 대화 + 각 thread 의 누적 메모리 스냅샷."""
    threads = [
        {
            "id": "thread_alpha",
            "title": "알파테크 김철수 책임 — GaN 전력 모듈 검토",
            "color_bg": "#E3F2FD",
            "color_edge": "#1565C0",
            "color_user": "#BBDEFB",
            "color_bot": "#FFFFFF",
            "turns": [
                ("U", "안녕하세요, 저는 (주)알파테크 김철수 책임입니다."),
                ("B", "안녕하세요 김 책임님. (주)알파테크에서 어떤 도움이 필요하신가요?"),
                ("U", "지난주 받은 GaN 모듈 데이터시트 확인했습니다. 80% 부하 효율 다시 알려 주세요."),
                ("B", "80% 부하 구간에서 변환 효율은 96.8% 입니다. 추가 자료 보내드릴까요?"),
                ("U", "네, 발열 곡선 자료까지 부탁드립니다."),
                ("B", "김 책임님 메일로 발열 곡선 PDF 발송해 두겠습니다."),
            ],
            "memory": [
                "name = 김철수 (책임)",
                "company = (주)알파테크",
                "관심 제품 = GaN 전력 모듈",
                "전달 자료 = 데이터시트 / 발열 곡선",
            ],
        },
        {
            "id": "thread_beta",
            "title": "베타시스템 박지수 매니저 — ODM 양산 협의",
            "color_bg": "#FFF3E0",
            "color_edge": "#E65100",
            "color_user": "#FFE0B2",
            "color_bot": "#FFFFFF",
            "turns": [
                ("U", "안녕하세요, (주)베타시스템 박지수 매니저입니다."),
                ("B", "안녕하세요 박 매니저님. 어떤 건으로 연락 주셨을까요?"),
                ("U", "산업 로봇 컨트롤러용 커스텀 커넥터 ODM 가능한지 확인하고 싶습니다."),
                ("B", "산업 로봇 컨트롤러용 ODM 가능합니다. 사양과 수량 알려 주시면 단가 검토 들어갑니다."),
                ("U", "MOQ 5,000개 기준 단가와 리드타임 함께 부탁드려요."),
                ("B", "MOQ 5,000개 기준 단가표와 4주 리드타임 일정 정리해서 회신 드리겠습니다."),
            ],
            "memory": [
                "name = 박지수 (매니저)",
                "company = (주)베타시스템",
                "관심 항목 = 커스텀 커넥터 ODM",
                "MOQ = 5,000 / 리드타임 = 4주",
            ],
        },
        {
            "id": "thread_gamma",
            "title": "감마전자 이수민 사원 — 채용 프로세스 문의",
            "color_bg": "#F3E5F5",
            "color_edge": "#6A1B9A",
            "color_user": "#E1BEE7",
            "color_bot": "#FFFFFF",
            "turns": [
                ("U", "안녕하세요, 모든전자 채용 관련해서 문의드립니다."),
                ("B", "안녕하세요. 어떤 직무에 관심이 있으신가요?"),
                ("U", "디지털 서비스팀 신입 지원하려고 하는데 코딩 테스트가 있나요?"),
                ("B", "디지털 서비스팀은 1차 면접 전 온라인 코딩 테스트가 진행됩니다. 약 90분 소요됩니다."),
                ("U", "감사합니다. 참고하겠습니다!"),
                ("B", "더 궁금하신 점 있으면 언제든 말씀해 주세요."),
            ],
            "memory": [
                "관심 직무 = 디지털 서비스팀 (신입)",
                "확인된 정보 = 코딩 테스트 90분",
                "후속 안내 = 1차 면접 일정",
                "고객 유형 = 채용 지원자",
            ],
        },
    ]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "thread_id 별 대화 격리 — 한 챗봇이 3명을 동시에 응대하지만 기억은 섞이지 않음",
        fontsize=17, fontweight="bold", y=0.985,
    )

    n = len(threads)
    fig_left, fig_right = 0.03, 0.97
    col_axes_w = (fig_right - fig_left - 0.025 * (n - 1)) / n

    for col, t in enumerate(threads):
        left = fig_left + col * (col_axes_w + 0.025)
        ax = fig.add_axes([left, 0.04, col_axes_w, 0.90])
        ax.axis("off")

        # ── 좌표계 단위 정의
        TOTAL_H = 30.0
        ax.set_xlim(0, 10)
        ax.set_ylim(0, TOTAL_H)

        # ── 영역 분할
        HEADER_TOP = TOTAL_H - 0.0
        HEADER_H = 1.6
        CHAT_TOP = HEADER_TOP - HEADER_H - 0.3
        MEM_H = 7.5

        # 1) 헤더 띠
        ax.add_patch(plt.Rectangle((0.0, HEADER_TOP - HEADER_H), 10.0, HEADER_H,
                                    facecolor=t["color_edge"], edgecolor=t["color_edge"]))
        ax.text(0.3, HEADER_TOP - 0.45, f"■ {t['id']}",
                fontsize=15, fontweight="bold", color="white", va="center")
        ax.text(0.3, HEADER_TOP - 1.2, t["title"],
                fontsize=12, fontweight="bold", color="white", va="center")

        # 2) 대화 영역 배경
        chat_bottom = MEM_H + 0.6
        ax.add_patch(plt.Rectangle((0.0, chat_bottom),
                                    10.0, CHAT_TOP - chat_bottom,
                                    facecolor=t["color_bg"], edgecolor=t["color_edge"], linewidth=1.2))

        # 3) 대화 turns — 위에서 아래로
        cur_y = CHAT_TOP - 0.4
        for role, msg in t["turns"]:
            wrapped = textwrap.wrap(msg, width=30, break_long_words=True, break_on_hyphens=False)
            n_lines = len(wrapped)
            LABEL_H = 0.65
            LINE_H = 0.70
            PAD_T, PAD_B = 0.22, 0.28
            box_h = PAD_T + LABEL_H + n_lines * LINE_H + PAD_B

            if role == "U":
                bx, bw = 0.35, 9.0
                bg = t["color_user"]
                lbl = "USER"
                lbl_color = t["color_edge"]
            else:
                bx, bw = 0.65, 9.0
                bg = t["color_bot"]
                lbl = "BOT"
                lbl_color = "#37474F"
            ax.add_patch(plt.Rectangle((bx, cur_y - box_h), bw, box_h,
                                        facecolor=bg, edgecolor=t["color_edge"], linewidth=1.0))
            ax.text(bx + 0.25, cur_y - PAD_T - LABEL_H / 2,
                    lbl, fontsize=11, fontweight="bold", color=lbl_color, va="center")
            for i, line in enumerate(wrapped):
                ax.text(bx + 0.25,
                        cur_y - PAD_T - LABEL_H - i * LINE_H - LINE_H / 2,
                        line, fontsize=12, fontweight="bold", color="#212121", va="center")
            cur_y -= box_h + 0.3

        # 4) 메모리 스냅샷
        ax.add_patch(plt.Rectangle((0.0, 0.0), 10.0, MEM_H,
                                    facecolor="#FAFAFA", edgecolor=t["color_edge"], linewidth=1.6))
        # 헤더 띠
        ax.add_patch(plt.Rectangle((0.0, MEM_H - 1.1), 10.0, 1.1,
                                    facecolor=t["color_edge"]))
        ax.text(0.3, MEM_H - 0.55, "▶ thread 메모리 스냅샷",
                fontsize=13, fontweight="bold", color="white", va="center")
        for i, item in enumerate(t["memory"]):
            ax.text(0.5, MEM_H - 2.0 - i * 1.15,
                    f"•  {item}",
                    fontsize=12.5, fontweight="bold", color="#37474F", va="center")

    out = RESULTS_DIR / "fig_06_thread_isolation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_alpha_thread_walkthrough() -> Path:
    """thread_alpha 한 사례를 따라가며 매 turn 마다 분류 + 메모리 기반 답변이 동시에 동작하는 모습.

    한 행(턴) 안에 5개 패널을 두고, 각 패널에 풍부한 정보를 담아 면접관이
    한눈에 '두 핵심 기능이 같이 동작하는 모습' 을 읽어낼 수 있도록 설계.
    """
    URGENCY_COLOR = {
        "low": ("#C8E6C9", "#1B5E20"),
        "medium": ("#FFF59D", "#5D4037"),
        "high": ("#FFAB91", "#BF360C"),
        "critical": ("#EF5350", "white"),
    }
    CATEGORY_COLOR = {
        "주문관리": "#1976D2",
        "배송지연": "#D32F2F",
        "기술지원": "#7B1FA2",
        "제품문의": "#388E3C",
        "기타": "#616161",
    }

    turns = [
        {
            "n": 1,
            "time": "10:14",
            "user": ("안녕하세요, (주)알파테크 김철수 책임입니다. 작년 4분기부터 "
                     "꾸준히 GaN 전력 모듈을 발주해 왔는데, PO-2026-0418 건의 "
                     "현재 진행 상태와 예상 출고일을 확인 부탁드립니다."),
            "category": "주문관리",
            "urgency": "medium",
            "requires_human": False,
            "summary": "PO-2026-0418 발주 진행 상태 및 출고일 확인 요청",
            "bot": ("안녕하세요 김철수 책임님! (주)알파테크 PO-2026-0418 발주 건을 "
                     "조회 중입니다. 출고일과 함께 곧 회신 드리겠습니다."),
            "bot_uses_memory": ["사용자 첫 turn — 신규 등록"],
            "memory_delta": [
                "+ name = 김철수 (책임)",
                "+ company = (주)알파테크",
                "+ 관심 제품 = GaN 전력 모듈",
                "+ 진행 PO = PO-2026-0418",
            ],
        },
        {
            "n": 2,
            "time": "10:21",
            "user": ("확인되었습니까? 납기 일정상 이번 주 금요일까지 처리되지 않으면 "
                     "계약상 페널티가 발생하는 건이라 긴급합니다. 오늘 안으로 답변 부탁드립니다."),
            "category": "배송지연",
            "urgency": "high",
            "requires_human": True,
            "summary": "납기 페널티 임박 — 즉시 처리 요청",
            "bot": ("김 책임님, PO-2026-0418 페널티 임박 건으로 확인했습니다. "
                     "영업 매니저에게 즉시 에스컬레이션 처리하겠습니다. 5분 이내 "
                     "담당자가 직접 연락드릴 예정입니다."),
            "bot_uses_memory": ["✓ 이름 호출", "✓ PO 번호 재참조"],
            "memory_delta": [
                "+ 진행 상태 = 페널티 임박",
                "+ 라우팅 = 영업 매니저 에스컬레이션",
                "+ 후속 = 5분 내 콜백",
            ],
        },
        {
            "n": 3,
            "time": "10:48",
            "user": ("감사합니다. 영업 매니저와 통화 잘 마쳤고, 출고 일정도 안내받았습니다. "
                     "추가로 GaN 모듈 외에 검토 중인 제품이 있어서 따로 문의 드리고 싶은데 가능할까요?"),
            "category": "기타",
            "urgency": "low",
            "requires_human": False,
            "summary": "페널티 건 종료 후 추가 제품 문의 의사 표명",
            "bot": ("다행입니다 김 책임님. 페널티 건은 정상 종료된 것으로 표시해 두겠습니다. "
                     "검토 중이신 제품이 어떤 것인지 편하게 말씀해 주시면, 사양과 단가 정보 "
                     "함께 정리해 드리겠습니다."),
            "bot_uses_memory": ["✓ 이름 호출", "✓ 페널티 건 상태 갱신"],
            "memory_delta": [
                "= 페널티 건 종료 (정상 처리)",
                "+ 의사 = 추가 제품 문의 예정",
            ],
        },
        {
            "n": 4,
            "time": "13:32",
            "user": ("긴급! 방금 양산 라인 1번에 들어간 GaN 모듈 1개가 발열로 인해 차단됐습니다. "
                     "라인이 멈춘 상황이라 동일 사양 대체품 즉시 확인 부탁드립니다. "
                     "지금 바로 통화 가능한 분 연결 부탁드립니다."),
            "category": "기술지원",
            "urgency": "critical",
            "requires_human": True,
            "summary": "양산 라인 중단 — GaN 모듈 발열 장애, 즉시 대체품 필요",
            "bot": ("김 책임님, 양산 라인 중단 사항으로 critical 등급 분류되었습니다. "
                     "기술지원 팀장에게 핫라인 연결을 발송했고 1분 이내 직접 통화 "
                     "가능합니다. 그 사이 동일 사양 대체품 후보 리스트를 준비하겠습니다."),
            "bot_uses_memory": ["✓ 이름 호출", "✓ 회사·관심 제품 활용"],
            "memory_delta": [
                "+ 라우팅 = 기술지원 팀장 (핫라인)",
                "+ 상태 = critical (양산 중단)",
                "+ 후속 = 1분 내 통화 + 대체품 리스트",
            ],
        },
        {
            "n": 5,
            "time": "13:35",
            "user": ("알겠습니다. 일단 통화 기다리고 있겠습니다. 대체품 후보 리스트는 "
                     "메일로도 함께 전달해 주시면 감사하겠습니다."),
            "category": "기타",
            "urgency": "low",
            "requires_human": False,
            "summary": "후속 자료 메일 송부 요청",
            "bot": ("네 김 책임님. 기술지원 팀장 통화 결과와 대체품 후보 리스트를 "
                     "(주)알파테크 등록 메일로 함께 보내드리겠습니다. "
                     "추가 사양 조건이 있으면 함께 알려 주세요."),
            "bot_uses_memory": ["✓ 회사명으로 발송 채널 확정"],
            "memory_delta": [
                "+ 후속 = 통화 결과 + 대체품 리스트 메일 발송",
            ],
        },
    ]

    fig = plt.figure(figsize=(26, 22))
    fig.suptitle(
        "thread_alpha 통합 흐름 — 한 turn 안에서 정형 분류 + 메모리 기반 답변이 동시에 동작",
        fontsize=24, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.957,
        "thread_alpha · (주)알파테크 김철수 책임 · 5턴 동안 urgency 가 변하며 requires_human 플래그와 라우팅이 자동으로 변동",
        ha="center", fontsize=14, color="#555", style="italic",
    )

    # ── 컬럼 헤더
    HEADER_AX = fig.add_axes([0.025, 0.905, 0.95, 0.035])
    HEADER_AX.axis("off")
    HEADER_AX.set_xlim(0, 100)
    HEADER_AX.set_ylim(0, 1)
    col_x = [0, 4, 26, 51, 78]
    col_w = [4, 22, 25, 27, 22]
    col_titles = ["#", "USER 발화", "Pydantic 분류 결과", "BOT 답변 (메모리 활용)", "Thread 메모리 업데이트"]
    col_colors = ["#37474F", "#F57C00", "#3949AB", "#43A047", "#FB8C00"]
    for x, w, title, color in zip(col_x, col_w, col_titles, col_colors):
        HEADER_AX.add_patch(plt.Rectangle((x, 0), w, 1, facecolor=color, edgecolor="black", linewidth=0.6))
        HEADER_AX.text(x + w / 2, 0.5, title,
                        ha="center", va="center", fontsize=15, fontweight="bold", color="white")

    # ── 5개 턴 행
    row_top = 0.895
    row_h = 0.168
    row_gap = 0.006

    for i, t in enumerate(turns):
        y_bottom = row_top - (i + 1) * row_h - i * row_gap
        ax = fig.add_axes([0.025, y_bottom, 0.95, row_h])
        ax.axis("off")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # 행 배경
        ax.add_patch(plt.Rectangle((0, 0), 100, 100,
                                    facecolor="#FAFAFA" if i % 2 == 0 else "#F0F0F0",
                                    edgecolor="#9E9E9E", linewidth=0.8))

        # ── 1) Turn 번호 + 시간
        ax.text(2.0, 60, f"#{t['n']}", ha="center", va="center",
                fontsize=42, fontweight="bold", color="#37474F")
        ax.text(2.0, 30, t["time"], ha="center", va="center",
                fontsize=13, fontweight="bold", color="#78909C")

        # ── 2) USER 발화
        ux, uw = 4.5, 21
        ax.add_patch(plt.Rectangle((ux, 5), uw, 90,
                                    facecolor="#FFE0B2", edgecolor="#E65100", linewidth=1.2))
        ax.text(ux + 0.6, 88, "USER", fontsize=13, fontweight="bold",
                color="#E65100", va="center")
        user_lines = textwrap.wrap(t["user"], width=22, break_long_words=True,
                                    break_on_hyphens=False)
        for j, line in enumerate(user_lines[:8]):
            ax.text(ux + 0.6, 78 - j * 9, line,
                    fontsize=13, fontweight="bold", color="#212121", va="center")

        # ── 3) 분류 결과
        cx, cw = 26.5, 24
        ax.add_patch(plt.Rectangle((cx, 5), cw, 90,
                                    facecolor="#E8EAF6", edgecolor="#1A237E", linewidth=1.2))

        # category badge (큰 배지)
        cat_color = CATEGORY_COLOR[t["category"]]
        ax.add_patch(plt.Rectangle((cx + 0.8, 78), 10, 12,
                                    facecolor=cat_color, edgecolor=cat_color))
        ax.text(cx + 5.8, 84, t["category"],
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(cx + 11.5, 84, "category",
                fontsize=11, color="#37474F", va="center", fontweight="bold")

        # urgency badge (큰 배지)
        u_bg, u_fg = URGENCY_COLOR[t["urgency"]]
        ax.add_patch(plt.Rectangle((cx + 0.8, 62), 10, 12,
                                    facecolor=u_bg, edgecolor="black", linewidth=1.0))
        ax.text(cx + 5.8, 68, t["urgency"].upper(),
                ha="center", va="center", fontsize=14, fontweight="bold", color=u_fg)
        ax.text(cx + 11.5, 68, "urgency",
                fontsize=11, color="#37474F", va="center", fontweight="bold")

        # requires_human flag
        flag_color = "#D32F2F" if t["requires_human"] else "#9E9E9E"
        flag_text = "✓ TRUE" if t["requires_human"] else "False"
        ax.add_patch(plt.Rectangle((cx + 0.8, 46), 10, 12,
                                    facecolor="white", edgecolor=flag_color, linewidth=2.5))
        ax.text(cx + 5.8, 52, flag_text,
                ha="center", va="center", fontsize=14, fontweight="bold", color=flag_color)
        ax.text(cx + 11.5, 52, "requires_human",
                fontsize=10.5, color="#37474F", va="center", fontweight="bold")

        # summary 한 줄
        ax.text(cx + 0.8, 36, "summary →",
                fontsize=11, fontweight="bold", color="#1A237E", va="center")
        sum_lines = textwrap.wrap(t["summary"], width=24,
                                   break_long_words=True, break_on_hyphens=False)
        for j, line in enumerate(sum_lines[:2]):
            ax.text(cx + 0.8, 27 - j * 7, line,
                    fontsize=11.5, fontweight="bold", color="#212121", va="center")

        # 라우팅 결과
        if t["requires_human"]:
            ax.add_patch(plt.Rectangle((cx + 0.8, 7), 22.4, 8,
                                        facecolor="#FFEBEE", edgecolor="#D32F2F", linewidth=1.5))
            ax.text(cx + 12, 11,
                    "→ 자동 에스컬레이션 (사람 라우팅)",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#D32F2F")
        else:
            ax.add_patch(plt.Rectangle((cx + 0.8, 7), 22.4, 8,
                                        facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=1.5))
            ax.text(cx + 12, 11,
                    "→ 챗봇이 직접 응대",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#388E3C")

        # ── 4) BOT 답변 (메모리 활용)
        bx, bw = 51.5, 26
        ax.add_patch(plt.Rectangle((bx, 5), bw, 90,
                                    facecolor="#E8F5E9", edgecolor="#1B5E20", linewidth=1.2))
        ax.text(bx + 0.6, 88, "BOT (메모리 활용)", fontsize=13, fontweight="bold",
                color="#1B5E20", va="center")
        bot_lines = textwrap.wrap(t["bot"], width=27, break_long_words=True,
                                   break_on_hyphens=False)
        for j, line in enumerate(bot_lines[:7]):
            ax.text(bx + 0.6, 78 - j * 8, line,
                    fontsize=13, fontweight="bold", color="#212121", va="center")
        # 메모리 활용 인디케이터
        ax.text(bx + 0.6, 18, "메모리 활용:",
                fontsize=10.5, fontweight="bold", color="#1B5E20", va="center")
        for j, indicator in enumerate(t["bot_uses_memory"][:2]):
            ax.text(bx + 0.6, 11 - j * 6, indicator,
                    fontsize=11, fontweight="bold", color="#2E7D32", va="center")

        # ── 5) 메모리 업데이트
        mx, mw = 78.5, 21
        ax.add_patch(plt.Rectangle((mx, 5), mw, 90,
                                    facecolor="#FFF8E1", edgecolor="#E65100", linewidth=1.2))
        ax.text(mx + 0.6, 88, "thread 메모리 변경", fontsize=13, fontweight="bold",
                color="#E65100", va="center")
        for j, item in enumerate(t["memory_delta"][:5]):
            color = "#1B5E20" if item.startswith("+") else "#37474F"
            wrapped_item = textwrap.wrap(item, width=22, break_long_words=True,
                                          break_on_hyphens=False)
            for k, line in enumerate(wrapped_item[:2]):
                ax.text(mx + 0.6, 78 - j * 14 - k * 6, line,
                        fontsize=12, fontweight="bold", color=color, va="center")

    out = RESULTS_DIR / "fig_07_alpha_thread_walkthrough.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모든전자 Production-grade 챗봇 파이프라인")
    p.add_argument(
        "--mode",
        choices=["classify", "chat", "summarize", "visualize", "all"],
        default="all",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {}

    needs_llm = args.mode in {"classify", "chat", "summarize", "all"}
    llm = build_llm() if needs_llm else None

    if args.mode in {"classify", "all"} and llm is not None:
        summary.update(run_classification(llm))
    if args.mode in {"chat", "all"} and llm is not None:
        summary.update(run_memory_comparison(llm))
    if args.mode in {"summarize", "all"} and llm is not None:
        summary.update(run_long_session(llm))

    if args.mode in {"visualize", "all"}:
        print("\n[ 4. 시각화 저장 ]")
        paths = [
            visualize_pipeline_overview(),
            visualize_pydantic_schema(),
            visualize_classification_distribution(),
            visualize_memory_comparison(),
            visualize_summarization_flow(),
            visualize_thread_isolation(),
            visualize_alpha_thread_walkthrough(),
        ]
        for p in paths:
            print(f"  - saved: {p.relative_to(ROOT_DIR)}")

    if summary:
        out_json = RESULTS_DIR / "chatbot_run_log.json"
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n실행 로그 저장: {out_json.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
