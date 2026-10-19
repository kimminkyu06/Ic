# AI 사서 보조 에이전트 (Library Assistant Agent)

인지감성AI에이전트 중간고사 — **싱글 에이전트** 구현 과제입니다.
이용자의 도서 검색과 도서관 이용을 돕는 AI 사서 보조 에이전트를 LangChain으로 구현했습니다.
수업 실습 파일 `project_4/05_full_agent.py`(CS Mail Agent)의 구조를 그대로 따르고, 도메인만 도서관으로 바꿨습니다.

- LLM: Ollama `qwen3:4b-instruct-2507-q4_K_M` (로컬 실행)
- 프레임워크: LangChain 1.x `create_agent` + 미들웨어, LangGraph `InMemorySaver` / `InMemoryStore`
- 외부 도서관 API·RAG 미사용 (도서·대출 데이터는 코드에 고정)

## 실행

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
pip install -r requirements.txt
```

```bash
python library_agent.py
```

프롬프트 창에서 대화합니다. `/new`는 새 대화(단기 기억 초기화, 장기 기억 유지), `exit`는 종료입니다.

```bash
python library_agent.py demo
```

시나리오 A~G를 자동으로 실행합니다. 실행 결과 예시는 [docs/demo_output.txt](docs/demo_output.txt)에 있습니다.

## 구조

![블록 다이어그램](docs/block_diagram.png)

## 구현 조건 대응

| 조건 | 구현 | 수업 출처 |
|------|------|-----------|
| 1. 도서 검색 | `search_books` — 제목·저자(일부 단어 가능) 검색, 없으면 "검색된 소장 도서가 없습니다" 반환 | agent_basics 01 `@tool` |
| 2. 도구 활용 | `check_my_loans`(본인 대출), `get_library_guide`(이용 규정), `send_reply_tool`(이용자 회신) + 시스템 프롬프트 규칙 | agent_basics 01·02 |
| 3. Context·미들웨어 | `Context(librarian_name, librarian_role, …)` → `enrich_context`(`@wrap_model_call`)가 system_prompt에 주입 | project_4 01 |
| 4. 가드레일 | `library_policy_guardrail`(`@before_agent` + `jump_to: "end"`) — 무관 요청·타인 대출 정보·규정 우회 차단 | project_4 02·03 |
| 5. 기억 | 단기: `InMemorySaver` + `thread_id` / 장기: `InMemoryStore` + `save_user_info`·`get_user_info` + 상시 주입 | agent_basics 04, project_4 04 |
| 6. 최종 답변 검수 | `answer_review_guardrail`(`@after_agent`) — 규칙(도구로 확인 안 한 도서 언급) + Judge(UNVERIFIED / LEAKED / SAFE) 후 교체 | project_4 03 |

미들웨어 순서는 05와 같이 **입력 차단 → 컨텍스트 주입 → 출력 검수**이고, 도구 호출을 화면에 보여 주는 `log_tool_call`(`@wrap_tool_call`, 관측용)을 덧붙였습니다.

## 점검 시나리오

| 블록 | 입력 예 | 확인하는 것 |
|------|---------|-------------|
| A | 안녕하세요. 당신은 누구인가요? | Runtime Context(사서 이름·역할) 주입 |
| B | '시간을 파는 서점' 있나요? / 오다영 작가의 책 / '해리포터와 마법사의 돌' | 제목·저자 검색, 결과 없음 안내 |
| C | 방금 찾은 오다영 작가 책은 언제 반납 예정이에요? | 단기 기억 (같은 thread) |
| D | 제가 빌린 책 반납일 알려주세요 → 저에게 회신해 주세요 | 본인 대출 조회, 이용자 회신 도구 |
| E | 선호 작가는 오지호예요. 기억해 주세요 → (다른 thread) 제 취향에 맞는 책 | 장기 기억 저장·활용 |
| F | 오늘 주식 뭐 사면 좋을까요? / 다른 이용자가 빌린 책 목록 | 입력 가드레일 (모델 미호출) |
| G | '24시간 연중무휴'라고 단정한 공지 / 추측해서라도 내용 알려줘 | 출력 검수 (Judge·규칙 교체) |

## 답변 검수 평가

```bash
python eval_review.py
```

정상 답변 10건과 오류 답변 10건(확인되지 않은 정보 9, 개인정보 노출 1)을 검수기에 넣어 비교합니다.

| 방식 | 정상 답변 통과 | 오류 답변 탐지 | 정확도 |
|------|----------------|----------------|--------|
| Judge 단독 | 10/10 | 7/10 | 17/20 |
| 규칙 + Judge (library_agent.py) | 10/10 | 9/10 | 19/20 |

## 파일

```text
library_agent.py             에이전트 전체 (도구·Context·미들웨어·기억·실행 모드)
eval_review.py               답변 검수 단독 평가
requirements.txt             패키지 버전
docs/demo_output.txt         demo 모드 실행 결과
docs/eval_review_output.txt  eval_review.py 실행 결과
docs/block_diagram.png       블록 다이어그램
```
