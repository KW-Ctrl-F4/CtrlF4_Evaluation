import os
import re
import json
import pandas as pd

# 각 문서 폴더들이 있는 루트 디렉터리
BASE = "dataset"
SAVE_MATCH_DETAILS = True   # per-doc 매칭 결과 저장 여부


###############################################
# 공용 함수
###############################################
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_safe(obj):
    """
    Recursively convert sets into sorted lists for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return sorted(list(obj))
    else:
        return obj


###############################################
# 1. loc 파싱 유틸
###############################################
def parse_clause_set(raw_clause) -> set[str]:
    """
    loc.clause 가 "3, 4" / "3" / "" / null 일 때
    → {"3","4"} 형태로 변환
    """
    if raw_clause is None:
        return set()
    s = str(raw_clause).strip()
    if not s:
        return set()
    # 콤마, 슬래시, 공백 등 기준으로 분리
    tokens = re.split(r"[,\u00b7/ ]+", s)
    return {t for t in tokens if t}


def parse_loc_gold(r):
    """
    골든라벨 risk_items 에서 loc 추출
    {
      "loc": {
        "article": 8,
        "clause": "3, 4",
        "page": 6
      }
    }

    article 가 "부칙 2조", "부칙 3조" 같이 문자열이어도
    숫자 부분(2, 3)을 뽑아서 평가에 포함되도록 보정.
    """
    loc = r.get("loc") or {}
    raw_article = loc.get("article")
    page = loc.get("page")
    clauses = parse_clause_set(loc.get("clause"))

    article = None
    if raw_article is not None:
        # 1차: 그대로 int 캐스팅 시도 (3, "3" 등)
        try:
            article = int(raw_article)
        except (ValueError, TypeError):
            # 2차: "부칙 2조", "제9조" 같은 문자열에서 숫자 부분만 추출
            m = re.search(r"(\d+)", str(raw_article))
            if m:
                article = int(m.group(1))

    return {
        "article": article,
        "clauses": clauses,
        "page": page,
    }


def parse_loc_llm(r):
    """
    GPT baseline (llm_raw.json) 의 risk_items 에서 loc 추출.
    골든라벨과 같은 스키마라고 가정.
    """
    loc = r.get("loc") or {}
    article = loc.get("article")
    page = loc.get("page")
    clauses = parse_clause_set(loc.get("clause"))

    if article is not None:
        try:
            article = int(article)
        except ValueError:
            article = None

    return {
        "article": article,
        "clauses": clauses,
        "page": page,
    }


def parse_loc_our(r):
    """
    우리 서비스 run-results 의 risk.items[*] 에서 article/항 추출.

    골든 구조에 최대한 맞추기 위해:
    - article:
        * "제 5 조", "제5조" 등에서 숫자 추출
        * 없으면 anchor.id 의 "art-5", "article-5" 등에서 숫자를 fallback
    - clause:
        * "제3항", "제 4 항" → "3", "4"
        * "1.6", "2.3" 같은 "번호.번호" 패턴도 그대로 clause 로 사용
        * 위 패턴들은 title / path / original_excerpt 전체에서 탐색
    """
    anchor = r.get("anchor") or {}
    title = str(anchor.get("title") or "")
    path = str(anchor.get("path") or "")
    aid = str(anchor.get("id") or "")
    excerpt = str(r.get("original_excerpt") or "")

    # 1) article: "제 5 조", "제5조" 등에서 숫자 뽑기
    article = None
    text_for_article = f"{title} {path}"
    m = re.search(r"제\s*(\d+)\s*조", text_for_article)
    if m:
        try:
            article = int(m.group(1))
        except ValueError:
            article = None

    # title/path 에서 못 뽑았으면 anchor.id 에서 fallback (art-5 / article-5 등)
    if article is None and aid:
        m_id = re.search(r"(?:art|article)[-_ ]*(\d+)", aid)
        if m_id:
            try:
                article = int(m_id.group(1))
            except ValueError:
                article = None

    # 2) clause:
    #    - "제3항" / "제 4 항" 패턴
    #    - "1.6", "2.3" 등 "번호.번호" 패턴 (gold 의 "1.6" 스타일과 정렬)
    clauses = set()
    text_for_clause = " ".join([title, path, excerpt])

    # (a) "제 3 항" 스타일
    clause_tokens = re.findall(r"제\s*(\d+)\s*항", text_for_clause)
    clauses.update(c for c in clause_tokens if c)

    # (b) "1.6", "2.3" 같은 "번호.번호" 패턴 전체를 clause 로 사용
    bullet_tokens = re.findall(r"\b\d+\.\d+\b", text_for_clause)
    clauses.update(bullet_tokens)

    return {
        "article": article,
        "clauses": clauses,
        "page": None,  # 페이지 정보는 현재 서비스에서 없으므로 일단 None
    }


###############################################
# 2. loc 기반 매칭
###############################################
def _normalize_clause_tokens(clause_set: set[str]) -> set[str]:
    """
    "1.6" vs "6" 처럼 표기만 다른 경우를 매칭시켜 주기 위한 정규화.

    - "1.6"  → "1.6" (원본도 유지)
              → 마지막 숫자 "6" 도 추가
    - "6"    → "6"
    """
    out = set()
    for t in clause_set or set():
        s = str(t)
        out.add(s)  # 원본 그대로도 유지
        # 끝에 붙은 숫자만 따로 하나 더 추가
        m = re.search(r"(\d+)$", s)
        if m:
            out.add(m.group(1))
    return out


def loc_match(g, p) -> bool:
    """
    기본 매칭 규칙 (이 함수는 그대로 두고, 실제 1:N 허용 로직은 count_loc_matches 에서 처리).

    g, p: {"article": int|None, "clauses": set[str], "page": ...}

    규칙:
      - article 가 둘 다 있고, 값이 같을 것
      - clause:
          * 둘 다 비어있지 않으면 교집합이 있을 것
            (단, "1.6" vs "6" 같이 표기만 다른 경우도 같은 것으로 처리)
          * 둘 중 하나라도 비어있으면 article만 같아도 매칭
            (조 단위 리스크라고 보고 인정)
    """
    ga, pa = g.get("article"), p.get("article")
    if ga is None or pa is None:
        return False
    if ga != pa:
        return False

    gc = g.get("clauses") or set()
    pc = p.get("clauses") or set()

    if gc and pc:
        # 1차: 원본 토큰 그대로 교집합
        if not gc.isdisjoint(pc):
            return True
        # 2차: "1.6" vs "6" 같은 케이스를 위해 숫자 정규화 후 교집합
        ng = _normalize_clause_tokens(gc)
        np = _normalize_clause_tokens(pc)
        return not ng.isdisjoint(np)

    # 조 단위 리스크 등인 경우 → article만 같으면 OK
    return True


def _clause_match_only(g, p) -> bool:
    """
    article 같고, 둘 다 clause 가 있을 때 '항 단위'까지 맞는지 확인하는 함수.
    (조 단위 매칭은 여기서 제외)
    """
    ga, pa = g.get("article"), p.get("article")
    if ga is None or pa is None or ga != pa:
        return False

    gc = g.get("clauses") or set()
    pc = p.get("clauses") or set()
    if not gc or not pc:
        return False

    # 원본 토큰
    if not gc.isdisjoint(pc):
        return True

    # 정규화 후 ("1.6" vs "6" 등)
    ng = _normalize_clause_tokens(gc)
    np = _normalize_clause_tokens(pc)
    return not ng.isdisjoint(np)


def count_loc_matches(gold_locs, pred_locs, gold_items=None, pred_items=None):
    """
    매칭 전략 (조 단위 한 덩어리 vs 잘게 나눈 gold 문제 해소용):

    1단계: '항까지 맞는' 1:1 매칭
      - _clause_match_only(g, p)가 True 인 경우에만 사용
      - pred 는 used_pred_clause 에 한 번만 사용됨

    2단계: 남은 gold 에 대해서
      - 같은 article 이면서 pred 쪽 clause 가 비어있는(조 단위) 경우,
        pred 하나가 여러 gold 와 중복 매칭될 수 있게 허용
      - 이때는 used_pred_clause 를 사용하지 않음 (재사용 허용)

    이렇게 하면:
      - 조 전체를 하나의 risk item 으로 쓴 our/llm 이
        같은 조의 여러 gold 리스크를 동시에 커버했다고 인정 받을 수 있음.
    """
    used_pred_clause = set()  # 항 단위 1:1 매칭에만 사용
    matched = 0
    details = []

    for gi, g_loc in enumerate(gold_locs):
        best_pj = None
        match_stage = None  # "clause" or "article_only"

        # -----------------------------
        # 1단계: 항까지 맞는 1:1 매칭 시도
        # -----------------------------
        for pj, p_loc in enumerate(pred_locs):
            if pj in used_pred_clause:
                continue
            if _clause_match_only(g_loc, p_loc):
                best_pj = pj
                match_stage = "clause"
                break

        # -----------------------------
        # 2단계: 조 단위(pred 쪽 clause 없음) 매칭 허용
        #        pred 하나가 여러 gold 와 중복 매칭될 수 있음
        # -----------------------------
        if best_pj is None:
            ga = g_loc.get("article")
            if ga is not None:
                for pj, p_loc in enumerate(pred_locs):
                    pa = p_loc.get("article")
                    pc = p_loc.get("clauses") or set()
                    # 같은 조이고, pred 쪽 clause 가 비어 있으면
                    if pa == ga and not pc:
                        best_pj = pj
                        match_stage = "article_only"
                        break

        d = {
            "gold_idx": gi,
            "pred_idx": best_pj,
            "matched": best_pj is not None,
            "gold_loc": g_loc,
            "match_stage": match_stage,
        }

        # 디버깅용: 원본 정보도 포함
        if gold_items is not None and 0 <= gi < len(gold_items):
            d["gold_item"] = {
                "type": gold_items[gi].get("type"),
                "loc": gold_items[gi].get("loc"),
            }

        if best_pj is not None:
            p_loc = pred_locs[best_pj]
            # 1단계(항 단위) 매칭이면 pred 한 번만 사용
            if match_stage == "clause":
                used_pred_clause.add(best_pj)

            matched += 1
            d["pred_loc"] = p_loc
            if pred_items is not None and 0 <= best_pj < len(pred_items):
                pred_obj = pred_items[best_pj]
                d["pred_item"] = {
                    "type": pred_obj.get("type") or pred_obj.get("riskType"),
                    "severity": pred_obj.get("severity"),
                    "loc": pred_obj.get("loc"),
                    "anchor_title": (pred_obj.get("anchor") or {}).get("title"),
                }

        details.append(d)

    return matched, details


###############################################
# 3. 메인 파이프라인
###############################################
results = []

for doc_folder in sorted(os.listdir(BASE)):
    path = os.path.join(BASE, doc_folder)
    if not os.path.isdir(path):
        continue

    # 이미 risk_match_details.json 이 있으면 스킵
    if SAVE_MATCH_DETAILS:
        detail_path = os.path.join(path, "risk_match_details.json")
        if os.path.exists(detail_path):
            print(f"\n=== Skipping {doc_folder} (risk_match_details.json already exists) ===")
            continue

    print(f"\n=== Processing {doc_folder} ===")

    gold_path     = os.path.join(path, "gold.json")
    our_raw_path  = os.path.join(path, "our_raw.json")
    llm_raw_path  = os.path.join(path, "llm_raw.json")

    if not os.path.exists(gold_path):
        print("  gold.json 없음 → 스킵")
        continue
    if not os.path.exists(our_raw_path):
        print("  our_raw.json 없음 → 스킵")
        continue
    if not os.path.exists(llm_raw_path):
        print("  llm_raw.json 없음 → 스킵 (ChatGPT 결과 저장 필요)")
        continue

    gold_raw = load_json(gold_path)
    our_raw  = load_json(our_raw_path)
    llm_raw  = load_json(llm_raw_path)

    # 3-1. gold loc 리스트
    gold_items = gold_raw.get("risk_items", [])
    gold_locs = [parse_loc_gold(r) for r in gold_items]
    gold_locs = [loc for loc in gold_locs if loc["article"] is not None]
    n_gold = len(gold_locs)

    if n_gold == 0:
        print("  gold loc.article 없는 항목만 있음 → 스킵")
        continue

    # 3-2. our loc 리스트 + 절대 risk count
    our_items_raw = (
        our_raw.get("results", {})
        .get("risk", {})
        .get("items", [])
    )
    our_locs = [parse_loc_our(r) for r in our_items_raw]
    our_locs = [loc for loc in our_locs if loc["article"] is not None]
    n_our = len(our_locs)

    # 3-3. llm loc 리스트 + 절대 risk count
    llm_items_raw = llm_raw.get("risk_items", [])
    llm_locs = [parse_loc_llm(r) for r in llm_items_raw]
    llm_locs = [loc for loc in llm_locs if loc["article"] is not None]
    n_llm = len(llm_locs)

    # 3-4. 매칭
    matched_our, details_our = count_loc_matches(
        gold_locs, our_locs,
        gold_items=gold_items,
        pred_items=our_items_raw,
    )
    matched_llm, details_llm = count_loc_matches(
        gold_locs, llm_locs,
        gold_items=gold_items,
        pred_items=llm_items_raw,
    )

    recall_our = matched_our / n_gold
    recall_llm = matched_llm / n_gold

    print(f"  gold 개수      : {n_gold}")
    print(f"  우리 risk 개수 : {n_our}  (탐지 {matched_our}개, recall={recall_our:.3f})")
    print(f"  LLM  risk 개수 : {n_llm}  (탐지 {matched_llm}개, recall={recall_llm:.3f})")

    results.append({
        "doc": doc_folder,
        "gold_count": n_gold,
        "our_risk_count": n_our,
        "llm_risk_count": n_llm,
        "our_detected": matched_our,
        "llm_detected": matched_llm,
        "our_recall": recall_our,
        "llm_recall": recall_llm,
    })

    # 디버깅용 상세 결과 저장
    if SAVE_MATCH_DETAILS:
        detail_path = os.path.join(path, "risk_match_details.json")
        save_json(detail_path, json_safe({
            "gold_locs": gold_locs,
            "our_locs": our_locs,
            "llm_locs": llm_locs,
            "our": details_our,
            "llm": details_llm,
        }))

###############################################
# 4. CSV 저장
###############################################
df = pd.DataFrame(results)
df.to_csv("risk_detection_results.csv", index=False)
print("\n=== Completed! Saved to risk_detection_results.csv ===\n")
