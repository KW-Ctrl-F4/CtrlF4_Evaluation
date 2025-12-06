import os
import re
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

###############################################
# 0. 모델 로딩
###############################################
model = SentenceTransformer("all-MiniLM-L6-v2")

###############################################
# 공용 함수
###############################################
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

###############################################
# 문장 분리
###############################################
def _split_sentences(text: str):
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    parts = re.split(r"\n+", text)
    sents = []

    for part in parts:
        if not part.strip():
            continue
        subs = re.split(r"(?<=[\.!?])\s+", part)
        for s in subs:
            s = s.strip()
            if s:
                sents.append(s)

    if not sents:
        return [text.strip()]
    return sents

###############################################
# 첫 문장 추출 (QA 핵심 문장)
###############################################
def extract_first_sentence(text: str):
    if not isinstance(text, str):
        return ""
    sents = _split_sentences(text.strip())
    return sents[0] if sents else text.strip()

###############################################
# 1. 평가 지표 함수들
###############################################
def semantic_similarity(original, summary):
    return float(util.cos_sim(
        model.encode(original),
        model.encode(summary)
    ))

def hallucination_rate(original, summary, threshold: float = 0.5):
    ori_sents = _split_sentences(original)
    sum_sents = _split_sentences(summary)

    if not sum_sents or not ori_sents:
        return 0.0

    if len(sum_sents) == 1 and len(ori_sents) == 1:
        sim = semantic_similarity(original, summary)
        return 0.0 if sim >= threshold else 1.0

    ori_embs = model.encode(ori_sents, convert_to_tensor=True)
    sum_embs = model.encode(sum_sents, convert_to_tensor=True)

    cos = util.cos_sim(sum_embs, ori_embs)

    hallucinated = 0
    for i in range(len(sum_sents)):
        max_sim = float(cos[i].max())
        if max_sim < threshold:
            hallucinated += 1

    return hallucinated / len(sum_sents)

def anchor_relevance(original, anchor_sentences):
    ori_sents = _split_sentences(original)
    if not ori_sents:
        return 0.0

    cleaned = []
    for anc in anchor_sentences:
        if not isinstance(anc, str):
            continue
        s = anc.strip()
        if len(s) < 10 and not re.search(r"\d", s):
            continue
        cleaned.append(s)

    if not cleaned:
        return 0.0

    scores = []
    for anc in cleaned:
        anc_emb = model.encode(anc)
        best = max(float(util.cos_sim(anc_emb, model.encode(sent))) for sent in ori_sents)
        scores.append(best)

    return sum(scores) / len(scores)

def qa_similarity(q, a):
    return float(util.cos_sim(
        model.encode(q),
        model.encode(a)
    ))

def risk_count(risks):
    return len(risks)

def risk_original_similarity(original, risks):
    ori_sents = _split_sentences(original)
    if not ori_sents:
        return 0.0

    sims = []
    for r in risks:
        cand = r.get("anchor") or r.get("text", "")
        if not isinstance(cand, str) or not cand.strip():
            continue

        r_emb = model.encode(cand.strip())
        best = max(float(util.cos_sim(r_emb, model.encode(sent))) for sent in ori_sents)
        sims.append(best)

    if not sims:
        return 0.0
    return sum(sims) / len(sims)

###############################################
# 1-1. 우리쪽 리스크 평가 제외 여부 판별
###############################################
def is_invalid_for_risk(raw_our: dict) -> bool:
    """
    our_raw.json 기준으로 '리스크 평가에서 제외할 문서'인지 판별
    - risk.items 비어 있고
    - summarizer.by_clause 비었거나
    - verifier.reason 에 retry limit 있고
    - QA 답변이 '컨텍스트 범위 내에서는 ... 확인할 수 없습니다.' 류 템플릿일 때
    """
    res = (raw_our.get("results") or {})

    # risk.items
    risk_items = (res.get("risk") or {}).get("items") or []
    if risk_items:
        # 어쨌든 뭔가 찾긴 했으면 평가 대상
        return False

    # summarizer.by_clause
    summ = (res.get("summarizer") or {}).get("results") or {}
    by_clause = summ.get("by_clause") or []

    # verifier.reason
    verifier = (res.get("verifier") or {})
    verifier_reason = (verifier.get("reason") or "").lower()

    # qa.answer
    qa = (res.get("qa") or {})
    qa_answer = (qa.get("answer") or "").strip()

    failed_signals = (
        (not by_clause) or
        ("retry limit" in verifier_reason) or
        ("컨텍스트 범위 내에서는" in qa_answer)
    )

    return failed_signals

###############################################
# 2. 변환기 (our_raw.json → our.json)
###############################################
def normalize_our(raw):
    """
    our_raw.json → 공통 포맷으로 정규화
    """

    # ========== SUMMARY ==========
    summ_block = raw.get("results", {}).get("summarizer", {}) or {}
    summary_text = ""

    # 1) 워커 ok 여부 확인 후 summary 추출
    if isinstance(summ_block, dict) and summ_block.get("ok", False):
        summ_results = summ_block.get("results", {}) or {}
        full_doc = summ_results.get("full_document", {}) or {}
        by_clause = summ_results.get("by_clause") or []

        summary_text = (full_doc.get("summary", "") or "").strip()

        # 2) 에러 메시지/빈 문자열이면 요약 제거
        if summary_text:
            lower_s = summary_text.lower()
            if lower_s.startswith("error"):
                summary_text = ""

        # 3) by_clause가 비어 있으면 요약 실패로 간주 → 평가 제외
        if not by_clause:
            summary_text = ""

    # 4) 공통 포맷
    if summary_text:
        summary = [{
            "text": summary_text,
            "anchor": summary_text,
        }]
    else:
        summary = []

    # ========== QA ==========
    qa_block = raw.get("results", {}).get("qa", {}) or {}

    first_answer = extract_first_sentence(qa_block.get("answer", ""))

    anchors = qa_block.get("anchors", []) or []
    if not isinstance(anchors, list):
        anchors = [str(anchors)]

    # 문자열만 정리
    anchors = [
        str(a).strip()
        for a in anchors
        if isinstance(a, str)
    ]

    # 답변 첫 문장도 anchor로 추가 (중복 방지)
    if first_answer and first_answer not in anchors:
        anchors.append(first_answer)

    qa = [{
        "question": qa_block.get("question", "") or "",
        "answer": first_answer,
        "anchors": anchors,
    }]

    # ========== RISK ==========
    our_items_raw = (
        raw.get("results", {})
           .get("risk", {})
           .get("items", [])
    )

    risk_items = []
    for r in our_items_raw:
        raw_excerpt = r.get("original_excerpt")

        # original_excerpt가 리스트/기타 타입이어도 안전하게 문자열로
        if isinstance(raw_excerpt, list):
            raw_excerpt = " ".join(str(x) for x in raw_excerpt)
        elif isinstance(raw_excerpt, (int, float)):
            raw_excerpt = str(raw_excerpt)
        elif not isinstance(raw_excerpt, str):
            raw_excerpt = "" if raw_excerpt is None else str(raw_excerpt)

        excerpt = (raw_excerpt or "").strip()

        anchor_obj = r.get("anchor") or {}
        if isinstance(anchor_obj, list):
            # anchor가 리스트일 경우 첫 원소를 dict로 가정하거나 문자열로 합치기
            if anchor_obj and isinstance(anchor_obj[0], dict):
                anchor_obj = anchor_obj[0]
            else:
                anchor_obj = {"title": " ".join(str(x) for x in anchor_obj)}

        anchor_title = str(anchor_obj.get("title") or "").strip() if isinstance(anchor_obj, dict) else ""
        anchor_id = str(anchor_obj.get("id") or "").strip() if isinstance(anchor_obj, dict) else ""

        reason = r.get("reason")
        if isinstance(reason, list):
            reason = " ".join(str(x) for x in reason)
        elif isinstance(reason, (int, float)):
            reason = str(reason)
        elif not isinstance(reason, str):
            reason = "" if reason is None else str(reason)
        reason = reason.strip()

        # text: 설명 위주의 짧은 요약 (notes 역할)
        text = extract_first_sentence(reason or anchor_title or excerpt)

        # anchor: 원문 조항을 최우선으로!
        anchor_str = excerpt or anchor_title or reason or anchor_id

        risk_items.append({
            "text": text,
            "anchor": anchor_str,
        })

    return {
        "summary": summary,
        "qa": qa,
        "risk_items": risk_items,
    }

###############################################
# 3. 변환기 (llm_raw.json → llm.json)
###############################################
def normalize_llm(raw):
    """
    llm_raw.json → 공통 포맷으로 정규화
    raw가 dict인 케이스와 list인 케이스(요약 리스트만 있는 경우) 둘 다 처리
    """

    # ----- Summary -----
    if isinstance(raw, list):
        # 최상위가 summary 아이템 리스트라고 가정
        summary_items = raw
        qa_pairs = []
        risk_items_raw = []
    elif isinstance(raw, dict):
        summary_items = raw.get("summary", [])
        qa_pairs = raw.get("qa_pairs", [])
        risk_items_raw = raw.get("risk_items", [])
    else:
        summary_items = []
        qa_pairs = []
        risk_items_raw = []

    summary_text = ""
    if isinstance(summary_items, list):
        # summary_items가 [{"text": ...}, ...] 형식이라고 가정
        parts = []
        for item in summary_items:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")).strip())
            else:
                parts.append(str(item).strip())
        summary_text = " ".join(p for p in parts if p)
    elif isinstance(summary_items, str):
        summary_text = summary_items

    summary_text = summary_text.strip()
    if summary_text:
        summary = [{"text": summary_text, "anchor": summary_text}]
    else:
        summary = []

    # ----- QA -----
    qa_list = []
    for q in qa_pairs:
        if not isinstance(q, dict):
            continue

        first_answer = extract_first_sentence(q.get("answer", ""))

        anchors = []
        ev = q.get("evidence")
        if isinstance(ev, dict):
            if isinstance(ev.get("quote"), str):
                anchors.append(ev["quote"].strip())
            elif isinstance(ev.get("text"), str):
                anchors.append(ev["text"].strip())
        elif isinstance(ev, str):
            anchors.append(ev.strip())
        elif isinstance(ev, list):
            for e in ev:
                if isinstance(e, str):
                    anchors.append(e.strip())
                elif isinstance(e, dict):
                    if isinstance(e.get("quote"), str):
                        anchors.append(e["quote"].strip())
                    elif isinstance(e.get("text"), str):
                        anchors.append(e["text"].strip())

        qa_list.append({
            "question": q.get("question", ""),
            "answer": first_answer,
            "anchors": anchors + ([first_answer] if first_answer else []),
        })

    # ----- Risk -----
    risk_items = []
    for r in risk_items_raw:
        if not isinstance(r, dict):
            continue

        text = extract_first_sentence(
            r.get("notes", "") or r.get("text", "") or r.get("type", "")
        )

        anchor = r.get("evidence", "")
        if isinstance(anchor, dict):
            anchor = anchor.get("quote", "") or anchor.get("text", "")
        if isinstance(anchor, list):
            # 리스트면 이어 붙이기
            anchor = " ".join(str(x).strip() for x in anchor)
        if not isinstance(anchor, str):
            anchor = ""

        risk_items.append({"text": text, "anchor": anchor.strip()})

    return {
        "summary": summary,
        "qa": qa_list,
        "risk_items": risk_items
    }

###############################################
# 4. 메인 파이프라인
###############################################
BASE = "dataset"
results = []

for doc_folder in sorted(os.listdir(BASE)):
    path = os.path.join(BASE, doc_folder)
    if not os.path.isdir(path):
        continue

    print(f"\n=== Processing {doc_folder} ===")

    original_path = os.path.join(path, "original.txt")
    our_raw_path = os.path.join(path, "our_raw.json")
    llm_raw_path = os.path.join(path, "llm_raw.json")

    if not (os.path.exists(original_path) and os.path.exists(our_raw_path) and os.path.exists(llm_raw_path)):
        print("Required files missing. Skipping.")
        continue

    original = load_text(original_path)
    our_raw = load_json(our_raw_path)
    llm_raw = load_json(llm_raw_path)

    our = normalize_our(our_raw)
    llm = normalize_llm(llm_raw)

    # Summary
    our_summary = our["summary"][0]["text"].strip() if our["summary"] else ""
    llm_summary = llm["summary"][0]["text"].strip() if llm["summary"] else ""

    if our_summary:
        summary_sim_ours = semantic_similarity(original, our_summary)
        halluc_ours = hallucination_rate(original, our_summary)
    else:
        summary_sim_ours = float("nan")
        halluc_ours = float("nan")

    if llm_summary:
        summary_sim_llm = semantic_similarity(original, llm_summary)
        halluc_llm = hallucination_rate(original, llm_summary)
    else:
        summary_sim_llm = float("nan")
        halluc_llm = float("nan")

    # QA
    qa_anchor_ours = []
    qa_anchor_llm = []
    qa_match_ours = []
    qa_match_llm = []

    for qa in our["qa"]:
        qa_anchor_ours.append(anchor_relevance(original, qa["anchors"]))
        qa_match_ours.append(qa_similarity(qa["question"], qa["answer"]))

    for qa in llm["qa"]:
        qa_anchor_llm.append(anchor_relevance(original, qa["anchors"]))
        qa_match_llm.append(qa_similarity(qa["question"], qa["answer"]))

    # Risk
    # 우리쪽: 실패 케이스는 NaN으로 처리해서 평균에서 제외
    if is_invalid_for_risk(our_raw):
        risk_count_ours = float("nan")
        risk_ori_ours = float("nan")
    else:
        risk_count_ours = risk_count(our["risk_items"])
        risk_ori_ours = risk_original_similarity(original, our["risk_items"])

    # LLM 쪽은 그대로 평가 (원하면 비슷한 필터 함수 따로 만들 수 있음)
    risk_count_llm = risk_count(llm["risk_items"])
    risk_ori_llm = risk_original_similarity(original, llm["risk_items"])

    results.append({
        "doc": doc_folder,
        "summary_sim_ours": summary_sim_ours,
        "summary_sim_llm": summary_sim_llm,
        "halluc_ours": halluc_ours,
        "halluc_llm": halluc_llm,
        "qa_anchor_ours": sum(qa_anchor_ours) / len(qa_anchor_ours) if qa_anchor_ours else float("nan"),
        "qa_anchor_llm": sum(qa_anchor_llm) / len(qa_anchor_llm) if qa_anchor_llm else float("nan"),
        "qa_match_ours": sum(qa_match_ours) / len(qa_match_ours) if qa_match_ours else float("nan"),
        "qa_match_llm": sum(qa_match_llm) / len(qa_match_llm) if qa_match_llm else float("nan"),
        "risk_count_ours": risk_count_ours,
        "risk_count_llm": risk_count_llm,
        "risk_ori_ours": risk_ori_ours,
        "risk_ori_llm": risk_ori_llm,
    })

###############################################
# 5. 저장
###############################################
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("\n=== Completed! Saved to evaluation_results.csv ===\n")
