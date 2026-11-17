# Research Brief: Do LLMs beat rules and classical matchers for entity resolution?

*Compiled 2026-11-17. 13 sources read. Method: 10 angle searches plus targeted page extraction via Firecrawl (Bright Data `bdata` was not installed in this environment, so Discover relevance scores are not available).*

## Executive summary

For clean, structured person records, the published evidence agrees with this repo's KILLED-AT-4 verdict: when records share several clean fields in a common format, classical probabilistic linkers match or beat LLM-based linkers (fastLink 96.7, Splink 97.4, LLM-labelled fuzzylink 97.1 F1 on North Carolina voter records) [6]. LLMs win clearly when the matching signal is textual: names in different scripts, abbreviations, translations, free text. There the gap is enormous (Splink 3.7 vs fuzzylink 66.6 F1 on multilingual names) [6]. The one head-to-head against a production rule baseline that LLMs win outright is OpenSanctions, where GPT-4o reached 98.95 F1 against the rule matcher's 91.33. That is also a name-heavy, cross-script task where the rule over-matches [4]. The strongest practical pattern in 2025–2026 work is hybrid: keep Fellegi–Sunter or a rule, send only the uncertain band to an LLM. One statistics-office study used that setup to halve the clerical-review gray zone (44.8% to 21.5% of pairs) [5]. Newer and larger models are not reliably better at matching [4], which is consistent with this repo's finding that claude-sonnet-5 was not significantly better than qwen2.5:14b.

## Key findings

### 1. Where LLMs beat trained or classical matchers

- Zero-shot GPT-4 beat fine-tuned RoBERTa and Ditto on 3 of 6 benchmarks (by 2.65–4.71 F1) without any training pairs, and lost on the other three by 0.33–4.49 F1 [1]. The wins were on e-commerce product data; the losses included the bibliographic sets, which are the more structured ones (DBLP-Scholar: Ditto 94.31 vs GPT-4 89.82) [1].
- The LLM advantage is largest when no in-domain training data exists: fine-tuned PLMs transferred to unseen WDC products dropped 36–56 F1, and GPT-4 beat the best transferred PLM by 40–68 F1 [1].
- On OpenSanctions Pairs, every frontier LLM tested beat the rule-based `nomenklatura` baseline (F1 91.33, precision 84.46, recall 99.42). GPT-4o scored 98.95, GPT-5.2 Pro 98.75, DeepSeek-R1-Distill-Qwen-14B 97.76, Claude 3.7 Sonnet 97.50 [4].
- On text-heavy linkage (multilingual names, census occupation strings, bill summaries), classical linkers collapse: fastLink/Splink F1 of 26.2 on 1940 census occupations to OCC1950 titles vs 65.9 for the LLM-labelled fuzzylink [6].

### 2. Where rules and classical linkers hold their ground

- On North Carolina voter records, "where records usually agree on name and birth year, every method links nearly every voter correctly". The spread between fastLink, Splink, fuzzylink and an embedding method was 96.7–97.5 F1 [6]. The author's conclusion: "fastLink and Splink remain strong choices when records share several clean information fields in a common format" [6].
- A retrieve-and-rerank method with no LLM (EnsembleLink) beat fuzzylink on city, candidate-voter and organization matching. Adding a Qwen3-8B reranker *lowered* accuracy on all three (e.g. 0.990 to 0.959 on candidate-voter) and only helped on multilingual party names, which needed world knowledge (0.843 to 0.894) [7].
- In one bibliographic error analysis, 11 of GPT-4's 26 false negatives were attributed to year discrepancies [1]. On OpenSanctions, LLM errors concentrate on transliteration and on "over-weighting of minor data-entry noise (off-by-one dates or identifiers) into spurious negatives" [4]. Both point at the kind of field (years, dates) that carries the signal in census linkage.

### 3. Hard negatives

- When near-duplicate distractors were added to standard benchmarks, every pairwise matcher lost precision. On Walmart-Amazon the false-positive increase column reads 66.7 for GPT-4 in-context learning and 128 for fine-tuned RoBERTa-large [10]. LLMs degrade less than PLMs here, but they still degrade. This repo's homonym negatives (same first name and patronym, birth within one year) are exactly this kind of distractor.

### 4. Pairwise vs. selecting among candidates

- Asking an LLM to pick the match from a candidate set ("selecting") beat independent yes/no pairwise matching by an average of 16.02 F1, at less than half the cost [3]. The ComEM pipeline puts a cheap model (Flan-T5-XL) in front to rank, then sends the top 4 to a stronger LLM, for 2–18 F1 over single strategies [3]. Census linkage is naturally one-to-many (one 1835 record, several plausible 1840 homonyms), so a pairwise test is the LLM's weakest format.

### 5. Hybrid use in official statistics

- On 1,000 Canadian business-record pairs, Llama 3 8B reached F1 0.87 vs 0.73 for Jaro–Winkler, at roughly 52 GPU-minutes vs under a second [5].
- Routing only the gray zone (prior match probability 0.35–0.85) to the LLM and combining its answer by Bayesian update raised overall F1 from 0.74 to 0.87. The gray zone shrank from 44.8% to 21.5% of pairs [5]. The LLM was weakest near prior 0.50, "because the data fundamentally lacked predictive power" [5].

### 6. Cost and throughput

- In a vendor test on 92,175 person records, an LLM cost about $10 per 10,000 records ($90–100 extrapolated for the full set). The purpose-built engine resolved all of them in 131 seconds at near-zero marginal cost. Accuracy was not measured [8].
- A BERT-sized Ditto model runs at 862,001 tokens/s, orders of magnitude above prompted LLMs [2]. Fine-tuning small models (AnyMatch, 124M–1.3B parameters) matched GPT-4 prompting on cross-dataset matching (87.5 vs 87.4 mean F1) [2].

### 7. Scoring and calibration (bears on AUC)

- Verbalized confidence lacks granularity: an ICLR 2026 paper reports that its method ranks "positives above negatives even among instances with a verbalized confidence of 1", which plain verbalized scores cannot do [13]. ComEM instead scores pairs from token probabilities, `1 + p(Yes)` or `1 - p(No)` [3].
- Repo observation (not from the literature): `src/score.py` asks for a verbalized probability, single sample. In `outputs/`, qwen2.5:14b gave 0.95 to 102 of 200 pairs and used 11 distinct values; claude-sonnet-5 used 24; the B2 rule produced 63 distinct scores. Heavy ties cap achievable AUC, so part of the LLM deficit may be a scoring artifact rather than a reasoning one.

## Contradictions and uncertainty

- OpenSanctions [4] shows LLMs beating a rule by about 7.6 F1, while this repo and the NC voter benchmark [6] show no LLM advantage. I think the domains, not the methods, differ. OpenSanctions entities are names and aliases across scripts, and the rule there is tuned for recall (99.42) at the cost of precision. Census rows with clean birth year, parish and farm fields are the opposite regime.
- Peeters et al. [1] report GPT-4 matching fine-tuned PLMs, while the EDBT cross-dataset study [2] finds a 1.3B fine-tuned model ties GPT-4. Both are consistent with "frontier zero-shot is good, small fine-tuned is as good and far cheaper". They disagree only on emphasis.
- Model recency does not predict matching quality. Claude Opus 4.5 (95.45 F1) scored below Claude 3.7 Sonnet (97.50), and GPT-5.2 Pro (98.75) roughly tied GPT-4o (98.95) on the same task [4].
- The cost sources [8] and the Tilores post [9] are from ER vendors selling the non-LLM alternative. [8] measured no accuracy. Treat them as cost evidence only.

## Gaps

- No published head-to-head of an LLM against a hand-built rule on historical census person linkage turned up. The ICE-ID v1 preprint title promises "NARS against LLMs, & a ML Ensemble" [11], but the extracted text mentioned no LLM evaluation, and v2 moves baseline results to a companion methods paper I could not locate. Get that paper before claiming novelty.
- No study measuring LLM entity-matching AUC from verbalized scores was found. The calibration point in finding 7 rests on general-purpose calibration work [13] plus this repo's own output distribution.
- Nothing retrieved addresses patronymic naming or low-surname-variety populations directly. The historical-linkage literature [12] stresses false-match rates (20–70% for phonetic-only methods; 14–37% for one ML method) rather than balanced-pair AUC.
- Figures were extracted from pages by Firecrawl's model-based query mode, not copied from raw text. Spot-check any number before quoting it in a paper.

## Implications for this repo

The verdict stands, and it matches the literature for clean structured fields. Three caveats are worth stating in `RESULTS.md`, because each one biases against the LLM. The scoring is verbalized and coarse; token log-probabilities or k-sample averaging would give a fairer ranking. The format is pairwise rather than selecting among homonym candidates, which cost about 16 F1 in [3]. And a 1:1 balanced AUC does not measure the false-match rate at realistic candidate ratios, the metric historical linkage cares about [12]. The two experiments the evidence most supports next are the messy tail (OCR-garbled or free-text fields, where [5][6] show large LLM gains) and gray-zone triage on top of the B2 rule, as in [5].

## Sources

[1] Peeters et al. Entity Matching using Large Language Models (EDBT 2025). https://arxiv.org/html/2310.11244v4. Primary; main LLM-vs-PLM benchmark.
[2] A Deep Dive Into Cross-Dataset Entity Matching with Large and Small Language Models (EDBT 2025). https://openproceedings.org/2025/conf/edbt/paper-224.pdf. Primary; throughput and small-model results.
[3] Wang et al. Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching (COLING 2025). https://aclanthology.org/2025.coling-main.8.pdf. Primary; selecting strategy and ComEM.
[4] OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching (2026). https://arxiv.org/html/2603.11051v2. Primary; the only LLM-vs-production-rule head-to-head found.
[5] Ather. LLM-assisted record linkage: A framework for official statistics. Statistical Journal of the IAOS, 2026. https://journals.sagepub.com/doi/10.1177/18747655261422068. Primary; hybrid gray-zone design.
[6] Dasanaike. Pre-Trained Language Models as Zero-Shot Tools for Social Science Research (Oct 2026). https://arxiv.org/html/2610.19048v1. Primary; fastLink/Splink vs LLM across clean and messy tasks.
[7] Dasanaike. EnsembleLink: Accurate Record Linkage Without Training Data (2026). https://arxiv.org/html/2601.21138v1. Primary; LLM reranker helps only where world knowledge is needed.
[8] Sullivan (Senzing). I Pitted an LLM Against Senzing for Agentic Entity Resolution (July 2026). https://senzing.com/llm-vs-senzing-agentic-entity-resolution-tokenomics/. Vendor; cost and time only.
[9] Renwick (Tilores). Can LLMs be used for Entity Resolution? https://tilores.io/content/can-llms-be-used-for-entity-resolution/. Vendor opinion; low weight.
[10] Fine-tuning large language models with contrastive margin ranking loss for selective entity matching in product data integration. Advanced Engineering Informatics, 2025. https://www.sciencedirect.com/science/article/pii/S1474034625004318. Primary; hard-negative precision drop.
[11] de Carvalho et al. ICE-ID (v1 and v2), 2025. https://arxiv.org/html/2506.13792v1 and https://arxiv.org/html/2506.13792v2. Primary; dataset used in this repo, baselines in an unlocated companion paper.
[12] Historical Census Record Linkage (review article, PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC6200350/. Primary review; false-match rates in census linkage.
[13] Calibrating Verbalized Confidence with Self-Generated Distractors (ICLR 2026). https://proceedings.iclr.cc/paper_files/paper/2026/file/c50bf8e50041545841e28c0b052f76d0-Paper-Conference.pdf. Primary; granularity limits of verbalized confidence.
