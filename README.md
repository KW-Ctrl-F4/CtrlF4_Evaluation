## 📌 Project Overview

한국어 계약서 120개를 대상으로  
**GPT 5.1 only vs. Ctrl+F4 다중 에이전트 기반 계약서 분석 시스템(Ours)**  
의 성능을 정량 평가하기 위한 **공개 벤치마크 & 재현 코드 레포지토리**입니다.

---

## 📁 Repository Structure


```bash
CTRL4F_EVALUATION/
├── dataset/            # 120개 계약서 폴더 (PDF/텍스트 + our/llm 결과)
├── results/            # run_evaluation.py 결과 CSV
│
├── risk_eval/          # 독소조항 정답셋 기반 평가 전용 모듈
│   ├── dataset/        # 31개 계약서 폴더 (정답셋 포함)
│   ├── results/        # risk_eval 전용 결과 CSV
│   ├── run_risk_eval.py
│   └── requirements.txt
│
├── run_evaluation.py   # Summary / QA / Risk 전체 메트릭 계산 스크립트
├── requirements.txt
└── README.md
```

---

## 📁 Dataset Structure


총 10개 카테고리, 120개 계약서

- 근로계약서
- 임대차계약서
- 서비스 이용약관 / 앱 이용약관
- 프랜차이즈 계약서
- 라이선스 / 저작권 / 이용계약서
- NDA
- 개인정보 처리방침
-  위탁 용역 / 광고 계약서 등
---

## 📊 Metrics

주요 5가지 평가 지표 포함:

- 요약 의미 유사도 (Summary Similarity)

- 요약 환각 비율 (Hallucination Rate)

- QA 근거 적합도 (QA Anchor Relevance)

- QA 답변 유사도 (QA Answer Match)

- 독소조항 유사도 (Risk–Original Similarity)

추가로, 골든라벨이 존재하는 31개 문서에 대해:

- 문서당 독소조항 탐지 개수 (Risk Count per Doc)

- 독소조항 탐지율 (Risk Recall)
(탐지된 골든 리스크 수 / 전체 골든 리스크 수)

각 지표는 Sentence-BERT 및 커스텀 매칭 로직을 기반으로 합니다.

---

## 🧪 Reproduction
```
pip install -r requirements.txt
python run_evaluation.py --dataset dataset/ --out results/

cd risk_eval
python run_risk_eval.py
```

---

## 📈 Results Summary

### Summary / QA / Risk Similarity (120개 docs)
| Metric                   | GPT 5.1 | Ours      | Improvement       |
| ------------------------ | ------- | --------- | ----------------- |
| Summary Similarity       | 0.693   | **0.743** | **+7.2%**         |
| Hallucination Rate       | 0.022   | **0.000** | **-100%** (환각 제거) |
| QA Anchor Relevance      | 0.830   | **0.889** | **+7.1%**         |
| QA Answer Match          | 0.639   | **0.707** | **+10.6%**        |
| Risk–Original Similarity | 0.831   | **0.863** | **+3.8%**         |

### Risk Detection with Gold Labels (31 docs)
| Metric                         | GPT 5.1 | Ours       | Improvement                       |
| -------------------------- | ------- | ---------- | ------------------------ |
| **문서 평균 독소조항 탐지수**         | 7.17개   | **12.25개** | **5.08개 더 탐지 (~+70.9%)** |
| **독소조항 탐지율 (Risk Recall)** | 34.04%  | **65.96%** | **약 +94% 향상** (▲31.9%p)  |
