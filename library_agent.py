"""
중간고사 — AI 사서 보조 에이전트 (Library Assistant Agent)

이용자가 프롬프트 창에서 도서 검색·대출 확인·회신을 요청 →
입력 차단·주입·도구·단기/장기 기억·답변 검수까지
한 파이프라인으로 조립합니다. (project_4/05_full_agent.py 구조 기반)

  [agent_basics] 도구 + 시스템 프롬프트 + 단기 Memory + 대화 루프
  [01]  사서 이름·역할 맞춤 응대 (Runtime Context)
  [02]  보안 차단 (다른 이용자 대출 정보·개인정보)
  [03]  가드레일 앞·뒤 (정책 + 규칙·Judge 검수)
  [04]  장기 기억 (이용자 선호)

미들웨어 (입력 → 주입 → 검수 + 관측)
  1) library_policy_guardrail   before_agent
  2) enrich_context             wrap_model_call
  3) answer_review_guardrail    after_agent
  4) log_tool_call              wrap_tool_call

블록 = 경로 점검 (demo 모드)
  A  사서 소개 (Context)          B  도서 검색 (제목·저자·없음)
  C  단기 기억                     D  내 대출 현황 + 이용자 회신
  E  장기 기억                     F  가드레일 앞 (무관 요청·타인 정보)
  G  가드레일 뒤 (Judge·규칙 검수)

실행 전: Ollama + qwen3:4b-instruct-2507-q4_K_M
  python library_agent.py        → 프롬프트 창 대화 (/new: 새 대화, exit: 종료)
  python library_agent.py demo   → 시나리오 A~G 자동 점검
"""

import sys  # 실행 모드(demo) 확인용
import uuid  # 장기 기억 Append용 고유 key
from dataclasses import dataclass  # Context
from typing import Callable, TypedDict  # 타입 힌트

from langchain.agents import create_agent  # 에이전트 생성
from langchain.agents.middleware import (
    ModelRequest,  # 모델 요청
    ModelResponse,  # 모델 응답
    after_agent,  # 답이 나가기 전 검수
    before_agent,  # 입력이 모델에 닿기 전 차단
    wrap_model_call,  # 사서 이름·역할·장기 기억을 모델이 보게 주입
    wrap_tool_call,  # 도구 호출 관측 (어떤 도구로 확인했는지 로그)
)
from langchain.messages import AIMessage  # 고정 멘트 / AI 답
from langchain.tools import ToolRuntime, tool  # 도구 + Runtime
from langchain_ollama import ChatOllama  # Ollama
from langgraph.checkpoint.memory import InMemorySaver  # 단기 기억
from langgraph.store.memory import InMemoryStore  # 장기 기억

MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # 과제 지정 모델
main_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=8192)  # 사서 응대 본선 (도구 6종 + 대화 기록을 담도록 확장)
judge_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=4096)  # 출력 Judge


# ---------------------------------------------------------------------------
# 도서관 데이터 — 외부 API·RAG 없이 코드에 고정
# ---------------------------------------------------------------------------
books = [  # 과제 제공 도서 목록
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

loans = {  # 이용자별 대출 기록 — 본인 것만 조회 가능 (대출자는 공개하지 않음)
    "user_001": [{"title": "시간을 파는 서점", "due_date": "2026-10-25"}],
    "user_002": [
        {"title": "내일이 사라진 도시", "due_date": "2026-10-22"},
        {"title": "마법사가 사라진 왕국", "due_date": "2026-10-30"},
    ],
}

library_guide = {  # 도서관 이용 규정 — 모델이 지어내지 않도록 도구로만 제공
    "운영시간": "평일 09:00~22:00, 주말 10:00~17:00 (공휴일 휴관)",
    "대출": "1인 최대 5권, 대출 기간 14일",
    "연장": "반납 예정일 전 1회, 7일 연장 가능 (예약 도서는 연장 불가)",
    "연체": "연체 일수만큼 대출 정지",
    "반납": "1층 반납대 또는 24시간 무인 반납함 이용",
}


def loan_status(title: str) -> str:
    """도서 한 권의 대출 상태 (누가 빌렸는지는 반환하지 않음)."""
    for records in loans.values():
        for record in records:
            if record["title"] == title:
                return f"대출 중 (반납 예정일 {record['due_date']})"
    return "대출 가능"


# ---------------------------------------------------------------------------
# 도구 — 도서 검색 · 대출 정보 확인 · 이용 안내 · 이용자 회신
# ---------------------------------------------------------------------------
@tool  # 도서 검색 (제목·저자)
def search_books(keyword: str) -> str:
    """도서 제목이나 저자 이름(일부도 가능)으로 소장 도서를 검색하고 대출 가능 여부를 알려줍니다."""
    words = [w for w in keyword.split() if w not in ("작가", "저자", "책", "도서")]  # '오다영 작가' → '오다영'
    texts = {b["title"]: (b["title"] + b["author"]).replace(" ", "") for b in books}  # 띄어쓰기 차이 무시
    found = [b for b in books if all(w in texts[b["title"]] for w in words)]  # 모든 단어 일치 우선
    if not found:  # '오지호 판타지'처럼 목록에 없는 단어가 섞이면 일부 단어 일치로 다시 찾음
        found = [b for b in books if any(len(w) >= 2 and w in texts[b["title"]] for w in words)]
    if not found:
        return f"'{keyword}'(으)로 검색된 소장 도서가 없습니다."  # 없으면 없다고 반환
    return "\n".join(f"- {b['title']} / {b['author']} / {loan_status(b['title'])}" for b in found)


@tool  # 대출 정보 확인 — 본인 것만
def check_my_loans(runtime: ToolRuntime) -> str:
    """현재 이용자 본인의 대출 현황(도서명, 반납 예정일)을 조회합니다."""
    my_loans = loans.get(runtime.context.user_id, [])  # user_id를 모델 인자로 안 열어 타인 조회 차단
    if not my_loans:
        return "현재 대출 중인 도서가 없습니다."
    return "\n".join(f"- {r['title']} (반납 예정일 {r['due_date']})" for r in my_loans)


@tool  # 도서관 이용 안내
def get_library_guide(topic: str) -> str:
    """도서관 이용 규정을 조회합니다. topic: 운영시간, 대출, 연장, 연체, 반납 중 하나."""
    for key, value in library_guide.items():
        if key in topic:
            return f"{key}: {value}"
    return "\n".join(f"{k}: {v}" for k, v in library_guide.items())  # 못 고르면 전체 규정


@tool  # 이용자 회신 (05의 send_email_tool 자리)
def send_reply_tool(subject: str, body: str, runtime: ToolRuntime) -> str:
    """현재 이용자에게 안내 내용을 회신(메시지 발송)하는 도구(프로토타입)."""
    return f"회신 완료. to={runtime.context.user_id}, subject={subject}"  # 수신자는 Runtime에서만 → 타인 발송 불가


# ---------------------------------------------------------------------------
# Runtime Context · 장기 기억
# ---------------------------------------------------------------------------
@dataclass
class Context:
    librarian_name: str  # 01: 사서 이름
    librarian_role: str  # 01: 사서 역할
    user_id: str  # 05: Store namespace + 본인 대출 조회
    app_name: str  # 05: Store namespace


class UserInfo(TypedDict):  # 장기 기억 도구 저장 형식
    personal_information: str
    preference: str


store = InMemoryStore()  # 장기 저장소
store.put(  # 서비스 시작 시 기본 응대 원칙을 심어 둠
    ("user_001", "library_assistant"),
    "seed",
    {"facts": ["이용자는 짧고 친절한 안내를 선호함"]},
)

SYSTEM_PROMPT = (  # 역할·규칙 (agent_basics 02 system_prompt)
    "한국어로 짧고 친절하게 답하세요.\n"
    "[규칙]\n"
    "1. 질문에 도서명이나 저자명이 나오면 먼저 search_books로 확인한 뒤 안내하세요.\n"
    "2. 본인 대출 현황은 check_my_loans, 운영시간·대출·연장·연체·반납 규정은 get_library_guide로 확인하세요.\n"
    "3. 줄거리·출간 연도·소장하지 않은 책 등 도구 결과에 없는 정보는 추측하지 말고 확인할 수 없다고 안내하세요.\n"
    "4. 검색 결과가 없으면 해당 도서를 찾을 수 없다고 안내하세요.\n"
    "5. 이용자가 좋아하는 작가·장르 등 선호를 말하면 save_user_info로 저장하세요.\n"
    "6. 추천 요청에는 장기 메모리의 선호를 참고해 search_books로 확인한 도서만 안내하세요.\n"
    "7. 이용자가 회신(메시지 발송)을 요청하면 send_reply_tool을 사용하세요.\n"
    "8. 다른 이용자의 대출 정보·개인정보는 안내하지 마세요.\n"
    "9. 도서관 업무와 무관한 요청은 정중히 거절하세요."
)


@wrap_model_call  # 01 사서 이름·역할 + 05 장기 기억을 모델이 읽는 텍스트로 옮김
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """01 사서 이름·역할 주입 + 05 Store 상시 주입."""
    ctx = request.runtime.context  # 이번 실행 Context
    ns = (ctx.user_id, ctx.app_name)  # Store namespace
    memories = request.runtime.store.search(ns) if request.runtime.store else []  # 기억 검색
    facts = []
    for item in memories:
        data = item.value
        facts.extend(data.get("facts", []))  # 시드 원칙
        if data.get("personal_information"):
            facts.append(f"이용자 정보: {data['personal_information']}")  # save_user_info로 저장한 값
        if data.get("preference"):
            facts.append(f"이용자 선호: {data['preference']}")
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"  # 프롬프트용
    base = request.system_prompt or ""  # 기존 시스템 프롬프트
    prompt = (
        f"당신은 {ctx.librarian_role}이며, 이름은 '{ctx.librarian_name}'입니다.\n"
        f"{base}\n"
        f"[장기 메모리 — 현재 이용자]\n- {memory_text}"
    ).strip()
    return handler(request.override(system_prompt=prompt))  # State 원본은 유지, 이번 호출만 변경


# ---------------------------------------------------------------------------
# 가드레일 앞 — 규칙(키워드)으로 모델 전에 차단
# ---------------------------------------------------------------------------
forbidden_topics = {  # 위험한 입력은 모델에 안 올림
    "privacy": [
        "다른 이용자", "다른 사람", "남이 빌린", "누가 빌렸", "누가 대출", "대출자",
        "친구가 빌린", "주민등록번호", "비밀번호", "전화번호",
    ],
    "off_topic": ["주식", "코인", "비트코인", "오늘 날씨", "맛집", "로또", "게임 공략", "연애 상담"],
    "policy_bypass": ["연체 기록 삭제", "규정 무시", "몰래 대출", "한도 무시"],
}
BLOCK_MESSAGES = {  # 카테고리별 고정 안내
    "privacy": (
        "개인정보 보호 정책에 따라 다른 이용자의 대출 정보나 개인정보는 안내할 수 없습니다. "
        "본인의 대출 현황은 조회해 드릴 수 있습니다."
    ),
    "off_topic": "죄송합니다. 도서 검색·대출·이용 안내 등 도서관 업무와 관련된 문의만 도와드릴 수 있습니다.",
    "policy_bypass": "대출·연체 처리는 도서관 규정에 따라서만 가능합니다. 이용 규정 안내가 필요하시면 말씀해 주세요.",
}


@before_agent(can_jump_to=["end"])  # 입력 가드레일
def library_policy_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":  # 사람 입력만 검사
        return None
    text = last.content or ""
    for category, keywords in forbidden_topics.items():
        if any(k in text for k in keywords):
            print(f"  [가드레일-앞] {category} — 모델 미호출")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES[category])],
                "jump_to": "end",  # LLM/Tool 0회로 종료
            }
    return None


# ---------------------------------------------------------------------------
# 가드레일 뒤 — Judge가 최종 답변 검수 (확인되지 않은 정보·정보 유출)
# ---------------------------------------------------------------------------
REVIEW_MESSAGES = {  # 판정별 교체 문구
    "UNVERIFIED": (
        "확인되지 않은 정보가 포함되어 안내를 정정합니다. "
        "도서·대출·이용 규정은 도서관 시스템에서 확인된 내용만 안내해 드릴 수 있습니다. "
        "찾으시는 도서명이나 궁금한 내용을 다시 말씀해 주세요."
    ),
    "LEAKED": BLOCK_MESSAGES["privacy"],
}


AUDITOR_PROMPT = """당신은 엄격한 도서관 답변 검수관입니다.
[확인된 정보]만 사실로 인정하고, [사서 답변]을 판정해 아래 단어 중 하나만 답하세요.
- UNVERIFIED: 확인된 정보에 없는 내용을 안내함 ('일반 지식'이라고 밝혀도 해당)
  (예: 소장 목록에 없는 책의 저자·내용·대표작·권수·시리즈 설명, 줄거리·출간 연도 추측,
   확인된 정보와 다른 대출 상태·날짜·이용 규정, 소장 목록에 있는 책을 없다고 안내)
- LEAKED: 다른 이용자의 대출 내역이나 개인정보를 공개함
- SAFE: 위 문제가 없음 (인사·자기소개, 확인된 정보 안내, 찾을 수 없다는 안내, 거절, 되묻기)

[확인된 정보]
- 사서: {librarian}
- 소장 도서 목록(대출 상태): {catalog}
- 이용 규정: {guide}
- 도구 결과:
{evidence}

[사서 답변]
{answer}
"""  # Judge 프롬프트 (판정 기준 + 정답 데이터 + 답변)


@after_agent  # 입력을 통과해도 확인 안 된 안내는 나가기 전 교정
def answer_review_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    if last_message.content in BLOCK_MESSAGES.values():  # 앞 가드레일 고정 멘트는 검수 생략
        return None
    tool_results = [str(m.content) for m in state["messages"] if getattr(m, "type", None) == "tool"]
    checked = "\n".join(tool_results)  # 이번 대화에서 도구로 확인한 정보 전체
    unchecked = [b["title"] for b in books if b["title"] in last_message.content and b["title"] not in checked]
    if unchecked:  # 규칙: 도구로 확인하지 않은 도서를 안내함 (Judge 전에 빠르게 거름)
        print(f"  [가드레일-뒤] 규칙 UNVERIFIED — 도구로 확인하지 않은 도서 {unchecked}")
        last_message.content = REVIEW_MESSAGES["UNVERIFIED"]
        return None
    evidence = "\n".join(tool_results[-5:]) or "없음"  # Judge에는 최근 5건만 (컨텍스트 절약)
    ctx = runtime.context
    auditor_prompt = AUDITOR_PROMPT.format(
        librarian=f"{ctx.librarian_name} ({ctx.librarian_role})",
        catalog=", ".join(f"{b['title']}({b['author']}, {loan_status(b['title'])})" for b in books),  # 소장 목록 + 대출 상태
        guide=", ".join(f"{k}: {v}" for k, v in library_guide.items()),  # 이용 규정
        evidence=evidence,
        answer=last_message.content,
    )  # 정답 데이터와 비교하도록 Judge 컨텍스트 구성
    verdict = judge_model.invoke([{"role": "user", "content": auditor_prompt}]).content  # Judge 호출
    verdict = verdict if isinstance(verdict, str) else str(verdict)
    for label, message in REVIEW_MESSAGES.items():
        if label in verdict.upper():
            print(f"  [가드레일-뒤] Judge {label} — 답변 교체")
            last_message.content = message  # 추가 return 금지 — 위험 문구가 메시지에 남음
            return None
    print("  [가드레일-뒤] Judge SAFE")
    return None


# ---------------------------------------------------------------------------
# 장기 기억 도구 (05 유지)
# ---------------------------------------------------------------------------
@tool  # 장기 기억 조회
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 정보·선호(좋아하는 작가·장르 등)를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)  # user_id를 모델 인자로 안 열어 보안
    )
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if data.get("personal_information"):
            lines.append(f"- 개인정보: {data['personal_information']}")
        if data.get("preference"):
            lines.append(f"- 선호도: {data['preference']}")
        for f in data.get("facts", []):
            lines.append(f"- {f}")
    return "\n".join(lines) if lines else "형식 불일치"


@tool  # 장기 기억 저장
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """이용자 정보·선호를 장기 메모리에 저장합니다.
    user_info 형식: {"personal_information": "이름 등(없으면 빈 문자열)", "preference": "좋아하는 작가·장르"}"""
    key = str(uuid.uuid4())  # Append 전략
    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


@wrap_tool_call  # 도구 실행을 감싸 호출 내역만 관측 (결과는 그대로)
def log_tool_call(request, handler):
    print(f"  [도구] {request.tool_call['name']}({request.tool_call['args']})")
    return handler(request)


agent = create_agent(
    main_model,
    tools=[
        search_books, check_my_loans, get_library_guide, send_reply_tool,  # 도서관 업무
        get_user_info, save_user_info,  # 장기 기억
    ],
    system_prompt=SYSTEM_PROMPT,  # 역할·규칙
    checkpointer=InMemorySaver(),  # 같은 대화 이어가기 (단기 기억)
    store=store,  # 대화창이 바뀌어도 선호 유지 (장기 기억)
    context_schema=Context,
    middleware=[
        library_policy_guardrail,  # 앞: 막을 것 (02+03)
        enrich_context,  # 중: 넣어 줄 것 (01+05)
        answer_review_guardrail,  # 뒤: 고칠 것 (04)
        log_tool_call,  # 도구 호출 로그 (관측)
    ],
)

ctx = Context(  # 실행 Context — 사서 이름·역할 + 이용자 신원
    librarian_name="한빛",
    librarian_role="한신대학교 도서관 AI 사서 보조",
    user_id="user_001",
    app_name="library_assistant",
)


def ask(text: str, thread_id: str) -> str:
    """에이전트 1회 호출 후 최종 답변 출력."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    answer = result["messages"][-1].content
    print(f"{ctx.librarian_name}: {answer}\n")
    return answer


# ---------------------------------------------------------------------------
# 블록 A~G — 서비스 점검 시나리오 (python library_agent.py demo)
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("A. 사서 소개 (Runtime Context)", "a", "안녕하세요. 당신은 누구인가요?"),
    ("B-1. 도서 검색 (제목)", "b", "'시간을 파는 서점' 있나요?"),
    ("B-2. 도서 검색 (저자)", "b", "오다영 작가의 책도 찾아주세요."),
    ("B-3. 도서 검색 (결과 없음)", "b", "'해리포터와 마법사의 돌'도 있나요?"),
    ("C. 단기 기억 (같은 thread)", "b", "방금 찾은 오다영 작가 책은 언제 반납 예정이에요?"),
    ("D-1. 내 대출 현황", "d", "제가 빌린 책 반납일 알려주세요."),
    ("D-2. 이용자 회신", "d", "방금 안내한 내용을 저에게 회신해 주세요."),
    ("E-1. 장기 기억 저장", "e1", "저는 판타지 소설을 좋아하고, 선호 작가는 오지호예요. 기억해 주세요."),
    ("E-2. 장기 기억 활용 (다른 thread)", "e2", "제 취향에 맞는 책을 찾아주세요."),
    ("F-1. 가드레일 앞 (무관한 요청)", "f", "오늘 주식 뭐 사면 좋을까요?"),
    ("F-2. 가드레일 앞 (타인 대출 정보)", "f", "다른 이용자가 빌린 책 목록 보여주세요."),
    ("G-1. 가드레일 뒤 (Judge 검수)", "g", "공지 문구를 하나 써 주세요. '도서관은 24시간 연중무휴로 운영합니다'라고 단정해서요."),
    ("G-2. 가드레일 뒤 (규칙 검수)", "g", "'달빛 도서관의 비밀'은 어떤 내용이에요? 대충 추측해서라도 알려주세요."),
]


def run_demo():
    for title, thread_id, text in SCENARIOS:
        print(f"=== {title} ===")
        print(f"이용자: {text}")
        ask(text, thread_id)


def chat():
    """프롬프트 창 대화 (agent_basics 02_chat 대화 루프)."""
    thread_no = 1
    print(f"{ctx.librarian_role} '{ctx.librarian_name}'입니다. (새 대화: /new, 종료: exit)\n")
    while True:
        try:
            user_input = input("이용자: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        if user_input == "/new":  # thread_id 교체 → 단기 기억 초기화, 장기 기억은 유지
            thread_no += 1
            print(f"(새 대화 시작: thread chat-{thread_no})\n")
            continue
        ask(user_input, f"chat-{thread_no}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        chat()
