# Overlap Parameter Impact — Benchmark Comparison Report

**Topic:** Central Banks  
**Golden Dataset:** Central_banks (27 questions)  
**Chunk Size:** 1000 characters (for overlap-applicable strategies)  
**Top-K:** 5  
**Overlap Configurations Tested:**
- v1 — 100 character overlap (30% ratio)
- v2 — 200 character overlap (30% ratio)
- v3 — 300 character overlap (30% ratio)

**Chunking Strategies Evaluated:** 7 (recursive, overlapping, header, semantic, sentence, paragraph, fixed_size)  
**Retriever Techniques:** 4 (dense, hybrid, dense_rerank, hybrid_rerank)  
**Total Experiment Combinations per Version:** 28 (7 strategies x 4 techniques)  
**Source Files:**
- `benchmark_report_Central_banks(v1).xlsx` — 100 character overlap
- `benchmark_report_Central_banks(v2).xlsx` — 200 character overlap
- `benchmark_report_Central_banks(v3).xlsx` — 300 character overlap

---

## 1. Critical Finding: v1 and v2 Are Identical

All 28 strategy+technique combinations produce identical scores in v1 and v2. Every metric — Hit@K, MRR, Precision@K, nDCG@K, Recall@K, Context Relevance, Faithfulness, Answer Correctness, Redundancy, and Composite — is the same. The only difference between v1 and v2 is latency (execution time), which is not a retrieval quality metric.

This means increasing overlap from 100 to 200 characters had zero measurable effect on chunk content or retrieval quality for this dataset.

**All subsequent comparisons are therefore between v1/v2 (treated as one baseline) and v3.**

---

## 2. Experiment Configuration

### 2.1 Overlap-Applicable Strategies — Chunk Properties

#### recursive

| Version | Overlap Chars | Overlap % | Num Chunks | Avg Words | Std Words | Min Words | Max Words | Median Words | Corpus Boundary | Corpus Redundancy |
|---------|---------------|-----------|------------|-----------|-----------|-----------|-----------|--------------|-----------------|-------------------|
| v1/v2   | 100/200       | 30%       | 59         | 93.81     | 55.88     | 1         | 161       | 109          | 0.3559          | 0.5448            |
| v3      | 300           | 30%       | 64         | 99.39     | 54.24     | 1         | 164       | 124          | 0.3438          | 0.5611            |

#### overlapping

| Version | Overlap Chars | Overlap % | Num Chunks | Avg Words | Std Words | Min Words | Max Words | Median Words | Corpus Boundary | Corpus Redundancy |
|---------|---------------|-----------|------------|-----------|-----------|-----------|-----------|--------------|-----------------|-------------------|
| v1/v2   | 100/200       | 30%       | 60         | 96.27     | 58.62     | 1         | 163       | 115          | 0.3667          | 0.5447            |
| v3      | 300           | 30%       | 64         | 103.08    | 56.99     | 1         | 164       | 133.5        | 0.4062          | 0.5613            |

### 2.2 Overlap-Inapplicable Strategies — Chunk Properties (Identical Across All Versions)

| Strategy   | Chunk Size | Overlap % | Num Chunks | Avg Words |
|------------|------------|-----------|------------|-----------|
| header     | var        | 0%        | 24         | 223.12    |
| semantic   | dynamic    | dynamic   | 13         | 419.31    |
| sentence   | ~3 sent    | 0%        | 81         | 67.38     |
| paragraph  | var        | 0%        | 50         | 106.22    |
| fixed_size | 1000       | 0%        | 59         | 92.42     |

---

## 3. Composite Score Comparison — All 28 Combinations

### 3.1 Overlap-Sensitive Strategies (v3 differs from v1/v2)

| Strategy    | Technique      | Composite v1/v2 | Composite v3 | Delta (v1 to v3) | Direction |
|-------------|----------------|-----------------|--------------|-------------------|-----------|
| overlapping | dense          | 0.5990          | 0.6037       | +0.0047           | IMPROVED  |
| overlapping | hybrid         | 0.5656          | 0.5722       | +0.0066           | IMPROVED  |
| recursive   | hybrid         | 0.5688          | 0.5740       | +0.0052           | IMPROVED  |
| recursive   | dense          | 0.6034          | 0.5998       | -0.0036           | DEGRADED  |
| recursive   | dense_rerank   | 0.5157          | 0.5124       | -0.0033           | DEGRADED  |
| recursive   | hybrid_rerank  | 0.5157          | 0.5124       | -0.0033           | DEGRADED  |
| overlapping | dense_rerank   | 0.5456          | 0.5056       | -0.0400           | DEGRADED  |
| overlapping | hybrid_rerank  | 0.5456          | 0.5056       | -0.0400           | DEGRADED  |

### 3.2 Overlap-Immune Strategies (identical across all versions)

| Strategy   | Technique      | Composite (all versions) |
|------------|----------------|--------------------------|
| semantic   | hybrid         | 0.6043                   |
| header     | hybrid         | 0.5978                   |
| semantic   | dense          | 0.5958                   |
| paragraph  | hybrid         | 0.5936                   |
| header     | dense          | 0.5906                   |
| fixed_size | dense          | 0.5901                   |
| sentence   | dense          | 0.5894                   |
| paragraph  | dense          | 0.5890                   |
| fixed_size | hybrid         | 0.5781                   |
| sentence   | hybrid         | 0.5431                   |
| fixed_size | dense_rerank   | 0.5384                   |
| fixed_size | hybrid_rerank  | 0.5384                   |
| paragraph  | dense_rerank   | 0.5332                   |
| paragraph  | hybrid_rerank  | 0.5332                   |
| header     | dense_rerank   | 0.5237                   |
| header     | hybrid_rerank  | 0.5237                   |
| sentence   | dense_rerank   | 0.5188                   |
| sentence   | hybrid_rerank  | 0.5188                   |
| semantic   | dense_rerank   | 0.4293                   |
| semantic   | hybrid_rerank  | 0.4293                   |

---

## 4. Detailed Metric Breakdown — Overlap-Sensitive Strategies

### 4.1 recursive

#### recursive + dense

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8889 | 0.8889 | 0.0000  |
| MRR               | 0.8457 | 0.8333 | -0.0124 |
| Precision@K       | 0.2370 | 0.2370 | 0.0000  |
| nDCG@K            | 0.3827 | 0.3808 | -0.0019 |
| Recall@K          | 0.5049 | 0.5049 | 0.0000  |
| Context Relevance | 0.6068 | 0.6098 | +0.0030 |
| Faithfulness      | 0.6791 | 0.6857 | +0.0066 |
| Answer Correct.   | 0.5085 | 0.5044 | -0.0041 |
| Redundancy        | 0.6801 | 0.6953 | +0.0152 |
| Token Cost        | 875    | 885    | +10     |
| Latency (ms)      | 5.52   | 5.65   | +0.13   |
| **Composite**     | **0.6034** | **0.5998** | **-0.0036** |

#### recursive + hybrid

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8889 | 0.8889 | 0.0000  |
| MRR               | 0.7790 | 0.7778 | -0.0012 |
| Precision@K       | 0.2074 | 0.2222 | +0.0148 |
| nDCG@K            | 0.3387 | 0.3523 | +0.0136 |
| Recall@K          | 0.4925 | 0.4925 | 0.0000  |
| Context Relevance | 0.5893 | 0.5957 | +0.0064 |
| Faithfulness      | 0.6592 | 0.6686 | +0.0094 |
| Answer Correct.   | 0.5020 | 0.5022 | +0.0002 |
| Redundancy        | 0.6647 | 0.6775 | +0.0128 |
| Token Cost        | 850    | 846    | -4      |
| Latency (ms)      | 10.76  | 11.69  | +0.93   |
| **Composite**     | **0.5688** | **0.5740** | **+0.0052** |

#### recursive + dense_rerank

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.7778 | 0.7778 | 0.0000  |
| MRR               | 0.7346 | 0.7222 | -0.0124 |
| Precision@K       | 0.1852 | 0.1852 | 0.0000  |
| nDCG@K            | 0.3110 | 0.3060 | -0.0050 |
| Recall@K          | 0.4308 | 0.4400 | +0.0092 |
| Context Relevance | 0.5878 | 0.5899 | +0.0021 |
| Faithfulness      | 0.6543 | 0.6589 | +0.0046 |
| Answer Correct.   | 0.4990 | 0.4968 | -0.0022 |
| Redundancy        | 0.6557 | 0.6722 | +0.0165 |
| Token Cost        | 847    | 841    | -6      |
| Latency (ms)      | 1351.18 | 1276.72 | -74.46 |
| **Composite**     | **0.5157** | **0.5124** | **-0.0033** |

#### recursive + hybrid_rerank

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.7778 | 0.7778 | 0.0000  |
| MRR               | 0.7346 | 0.7222 | -0.0124 |
| Precision@K       | 0.1852 | 0.1852 | 0.0000  |
| nDCG@K            | 0.3110 | 0.3060 | -0.0050 |
| Recall@K          | 0.4308 | 0.4400 | +0.0092 |
| Context Relevance | 0.5878 | 0.5899 | +0.0021 |
| Faithfulness      | 0.6543 | 0.6589 | +0.0046 |
| Answer Correct.   | 0.4990 | 0.4968 | -0.0022 |
| Redundancy        | 0.6557 | 0.6722 | +0.0165 |
| Token Cost        | 847    | 841    | -6      |
| Latency (ms)      | 1256.44 | 1173.02 | -83.42 |
| **Composite**     | **0.5157** | **0.5124** | **-0.0033** |

### 4.2 overlapping

#### overlapping + dense

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8889 | 0.8889 | 0.0000  |
| MRR               | 0.8457 | 0.8519 | +0.0062 |
| Precision@K       | 0.2296 | 0.2370 | +0.0074 |
| nDCG@K            | 0.3793 | 0.3871 | +0.0078 |
| Recall@K          | 0.4872 | 0.4872 | 0.0000  |
| Context Relevance | 0.6082 | 0.6102 | +0.0020 |
| Faithfulness      | 0.6799 | 0.6809 | +0.0010 |
| Answer Correct.   | 0.5065 | 0.5058 | -0.0007 |
| Redundancy        | 0.6859 | 0.6921 | +0.0062 |
| Token Cost        | 924    | 917    | -7      |
| Latency (ms)      | 5.65   | 5.38   | -0.27   |
| **Composite**     | **0.5990** | **0.6037** | **+0.0047** |

#### overlapping + hybrid

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8889 | 0.8889 | 0.0000  |
| MRR               | 0.7778 | 0.7809 | +0.0031 |
| Precision@K       | 0.2074 | 0.2222 | +0.0148 |
| nDCG@K            | 0.3381 | 0.3523 | +0.0142 |
| Recall@K          | 0.4749 | 0.4749 | 0.0000  |
| Context Relevance | 0.5917 | 0.5958 | +0.0041 |
| Faithfulness      | 0.6598 | 0.6677 | +0.0079 |
| Answer Correct.   | 0.5006 | 0.5016 | +0.0010 |
| Redundancy        | 0.6687 | 0.6835 | +0.0148 |
| Token Cost        | 891    | 880    | -11     |
| Latency (ms)      | 11.21  | 11.67  | +0.46   |
| **Composite**     | **0.5656** | **0.5722** | **+0.0066** |

#### overlapping + dense_rerank

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8519 | 0.7407 | -0.1112 |
| MRR               | 0.7512 | 0.7222 | -0.0290 |
| Precision@K       | 0.2148 | 0.1926 | -0.0222 |
| nDCG@K            | 0.3322 | 0.3174 | -0.0148 |
| Recall@K          | 0.4422 | 0.4162 | -0.0260 |
| Context Relevance | 0.5891 | 0.5906 | +0.0015 |
| Faithfulness      | 0.6574 | 0.6573 | -0.0001 |
| Answer Correct.   | 0.5018 | 0.4966 | -0.0052 |
| Redundancy        | 0.6583 | 0.6695 | +0.0112 |
| Token Cost        | 874    | 870    | -4      |
| Latency (ms)      | 1360.74 | 1311.04 | -49.70 |
| **Composite**     | **0.5456** | **0.5056** | **-0.0400** |

#### overlapping + hybrid_rerank

| Metric            | v1/v2  | v3     | Delta   |
|-------------------|--------|--------|---------|
| Hit@K             | 0.8519 | 0.7407 | -0.1112 |
| MRR               | 0.7512 | 0.7222 | -0.0290 |
| Precision@K       | 0.2148 | 0.1926 | -0.0222 |
| nDCG@K            | 0.3322 | 0.3174 | -0.0148 |
| Recall@K          | 0.4422 | 0.4162 | -0.0260 |
| Context Relevance | 0.5891 | 0.5909 | +0.0018 |
| Faithfulness      | 0.6574 | 0.6575 | +0.0001 |
| Answer Correct.   | 0.5018 | 0.4966 | -0.0052 |
| Redundancy        | 0.6583 | 0.6703 | +0.0120 |
| Token Cost        | 874    | 870    | -4      |
| Latency (ms)      | 1288.85 | 1229.45 | -59.40 |
| **Composite**     | **0.5456** | **0.5056** | **-0.0400** |

---

## 5. Metric-Level Impact Analysis

### 5.1 Metrics Most Affected by Overlap Version Change (v1/v2 to v3)

| Metric      | Max Observed Delta | Strategy + Technique with Largest Change           |
|-------------|--------------------|----------------------------------------------------|
| Hit@K       | -0.1112            | overlapping + dense_rerank (0.8519 to 0.7407)      |
| Redundancy  | +0.0165            | recursive + dense_rerank (0.6557 to 0.6722)        |
| MRR         | -0.0290            | overlapping + dense_rerank (0.7512 to 0.7222)      |
| Recall@K    | -0.0260            | overlapping + dense_rerank (0.4422 to 0.4162)      |
| Precision@K | -0.0222            | overlapping + dense_rerank (0.2148 to 0.1926)      |
| Precision@K | +0.0148            | overlapping + hybrid (0.2074 to 0.2222)            |
| nDCG@K      | -0.0148            | overlapping + dense_rerank (0.3322 to 0.3174)      |
| nDCG@K      | +0.0142            | overlapping + hybrid (0.3381 to 0.3523)            |

### 5.2 Metrics Least Affected

| Metric            | Typical Delta Range  | Observation                                         |
|-------------------|----------------------|-----------------------------------------------------|
| Answer Correct.   | -0.0052 to +0.0010  | Negligible variation                                 |
| Context Relevance | -0.0000 to +0.0064  | Very low sensitivity                                 |
| Faithfulness      | -0.0001 to +0.0094  | Low sensitivity, tends to slightly improve in v3     |

---

## 6. Retriever Technique Interaction with Overlap

This is the most significant pattern in the Central Banks data: the direction of overlap impact depends on the retriever technique.

### 6.1 dense and hybrid — Tend to Improve with More Overlap (v3)

| Strategy    | Technique | Composite v1/v2 | Composite v3 | Delta   |
|-------------|-----------|-----------------|--------------|---------|
| overlapping | dense     | 0.5990          | 0.6037       | +0.0047 |
| overlapping | hybrid    | 0.5656          | 0.5722       | +0.0066 |
| recursive   | hybrid    | 0.5688          | 0.5740       | +0.0052 |
| recursive   | dense     | 0.6034          | 0.5998       | -0.0036 |

3 out of 4 dense/hybrid combos improve in v3. The exception is recursive+dense which drops slightly.

### 6.2 dense_rerank and hybrid_rerank — Degrade with More Overlap (v3)

| Strategy    | Technique      | Composite v1/v2 | Composite v3 | Delta   |
|-------------|----------------|-----------------|--------------|---------|
| overlapping | dense_rerank   | 0.5456          | 0.5056       | -0.0400 |
| overlapping | hybrid_rerank  | 0.5456          | 0.5056       | -0.0400 |
| recursive   | dense_rerank   | 0.5157          | 0.5124       | -0.0033 |
| recursive   | hybrid_rerank  | 0.5157          | 0.5124       | -0.0033 |

All 4 rerank combos degrade in v3. The overlapping strategy is hit hardest (-0.0400), driven primarily by Hit@K dropping from 0.8519 to 0.7407.

---

## 7. Chunk Property Changes (v1/v2 to v3)

| Property          | recursive (v1/v2 to v3) | overlapping (v1/v2 to v3) |
|-------------------|-------------------------|---------------------------|
| Num Chunks        | 59 to 64 (+5)           | 60 to 64 (+4)             |
| Avg Words         | 93.81 to 99.39 (+5.58)  | 96.27 to 103.08 (+6.81)   |
| Std Words         | 55.88 to 54.24 (-1.64)  | 58.62 to 56.99 (-1.63)    |
| Median Words      | 109 to 124 (+15)        | 115 to 133.5 (+18.5)      |
| Corpus Boundary   | 0.3559 to 0.3438 (-0.0121) | 0.3667 to 0.4062 (+0.0395) |
| Corpus Redundancy | 0.5448 to 0.5611 (+0.0163) | 0.5447 to 0.5613 (+0.0166) |

More overlap characters (v3) produces:
- More chunks (4-5 additional)
- Higher average words per chunk
- Higher median words per chunk
- Higher corpus redundancy (more shared text between chunks)

---

## 8. Strategy Sensitivity Ranking

Ranked by maximum absolute composite delta from v1/v2 to v3.

| Rank | Strategy    | Largest Absolute Delta | Technique               | Direction |
|------|-------------|------------------------|-------------------------|-----------|
| 1    | overlapping | 0.0400                 | dense_rerank/hybrid_rerank | DEGRADED  |
| 2    | overlapping | 0.0066                 | hybrid                  | IMPROVED  |
| 3    | recursive   | 0.0052                 | hybrid                  | IMPROVED  |
| 4    | overlapping | 0.0047                 | dense                   | IMPROVED  |
| 5    | recursive   | 0.0036                 | dense                   | DEGRADED  |
| 6    | recursive   | 0.0033                 | dense_rerank/hybrid_rerank | DEGRADED  |
| 7    | header      | 0.0000                 | (all)                   | NO CHANGE |
| 7    | semantic    | 0.0000                 | (all)                   | NO CHANGE |
| 7    | sentence    | 0.0000                 | (all)                   | NO CHANGE |
| 7    | paragraph   | 0.0000                 | (all)                   | NO CHANGE |
| 7    | fixed_size  | 0.0000                 | (all)                   | NO CHANGE |

---

## 9. Global Leaderboard — Top 10 Across All Versions

| Rank | Strategy    | Technique | Best Version   | Composite |
|------|-------------|-----------|----------------|-----------|
| 1    | semantic    | hybrid    | any (identical)| 0.6043    |
| 2    | overlapping | dense     | v3             | 0.6037    |
| 3    | recursive   | dense     | v1/v2          | 0.6034    |
| 4    | overlapping | dense     | v1/v2          | 0.5990    |
| 5    | header      | hybrid    | any (identical)| 0.5978    |
| 6    | semantic    | dense     | any (identical)| 0.5958    |
| 7    | paragraph   | hybrid    | any (identical)| 0.5936    |
| 8    | header      | dense     | any (identical)| 0.5906    |
| 9    | fixed_size  | dense     | any (identical)| 0.5901    |
| 10   | sentence    | dense     | any (identical)| 0.5894    |

---

## 10. Observations

### 10.1 v1 and v2 Produce Identical Results

Increasing overlap from 100 to 200 characters had no measurable effect on any metric for any strategy. This indicates that for the Central Banks document, the boundary between 100 and 200 character overlap does not cross a threshold that would change chunk content meaningfully enough to affect retrieval.

### 10.2 v3 Has Mixed Impact — Technique-Dependent

Unlike the Banking System dataset (where more overlap consistently improved scores), the Central Banks dataset shows a split:
- **dense and hybrid retrievers** generally benefit from more overlap (v3), with improved Precision@K, nDCG@K, and Faithfulness.
- **Rerank-based retrievers** degrade with more overlap (v3), with the overlapping strategy seeing a dramatic -0.1112 drop in Hit@K and -0.0400 in Composite.

### 10.3 Reranker Degradation in v3

The overlapping + rerank combinations lose Hit@K (0.8519 to 0.7407), meaning more queries fail to retrieve any relevant chunk in the top-K results after reranking. This suggests that increased overlap produces more similar chunks that confuse the reranker's scoring function.

### 10.4 dense_rerank and hybrid_rerank Convergence

As in the Banking System dataset, dense_rerank and hybrid_rerank produce identical metric scores for the same strategy and version, differing only in latency. This pattern holds across all versions and both overlap-sensitive strategies.

### 10.5 Overlap-Immune Strategies Dominate the Top of the Leaderboard

The overall best combination (semantic + hybrid, 0.6043) is unaffected by overlap changes. 6 of the top 10 positions in the global leaderboard are held by overlap-immune strategies, suggesting that for the Central Banks dataset, structural and semantic chunking approaches outperform overlap-dependent strategies in most cases.

---

## 11. Data Completeness Notes

- All 28 strategy+technique combinations were evaluated under each of the 3 version configurations, totaling 84 experiments.
- All 27 golden dataset questions were used for every experiment.
- v1 and v2 were verified to produce identical scores across all 28 combinations at both the Experiment Matrix and Heatmap sheet levels.
- The 5 overlap-immune strategies (header, semantic, sentence, paragraph, fixed_size) were verified to produce identical results across all 3 versions.
- dense_rerank and hybrid_rerank produce identical metric scores for the same strategy and version in all observed cases, differing only in latency.
