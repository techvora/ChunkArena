# Overlap Parameter Impact — Benchmark Comparison Report

**Topic:** Banking System  
**Golden Dataset:** Banking_system (22 questions)  
**Chunk Size:** 1000 characters (for overlap-applicable strategies)  
**Top-K:** 5  
**Overlap Configurations Tested:** 30% (100), 20% (200), 10% (300)  
**Chunking Strategies Evaluated:** 7 (recursive, overlapping, header, semantic, sentence, paragraph, fixed_size)  
**Retriever Techniques:** 4 (dense, hybrid, dense_rerank, hybrid_rerank)  
**Total Experiment Combinations per Config:** 28 (7 strategies x 4 techniques)  
**Source Files:**
- `benchmark_report_Banking_system(100_overlap).xlsx` — overlap at 30%
- `benchmark_report_Banking_system(200_overlap).xlsx` — overlap at 20%
- `benchmark_report_Banking_system(300_overlap).xlsx` — overlap at 10%

---

## 1. Experiment Configuration

### 1.1 Overlap-Applicable Strategies — Chunk Properties

These two strategies accept an overlap parameter and produce different chunk outputs per configuration.

#### recursive

| Config | Overlap % | Num Chunks | Avg Words | Std Words | Min Words | Max Words | Median Words | Corpus Boundary | Corpus Redundancy |
|--------|-----------|------------|-----------|-----------|-----------|-----------|--------------|-----------------|-------------------|
| 100    | 30%       | 48         | 96.40     | 49.03     | 1         | 161       | 102          | 0.3125          | 0.5676            |
| 200    | 20%       | 48         | 91.31     | 50.80     | 1         | 161       | 98           | 0.3125          | 0.5632            |
| 300    | 10%       | 47         | 87.30     | 52.44     | 1         | 161       | 97           | 0.3191          | 0.5550            |

#### overlapping

| Config | Overlap % | Num Chunks | Avg Words | Std Words | Min Words | Max Words | Median Words | Corpus Boundary | Corpus Redundancy |
|--------|-----------|------------|-----------|-----------|-----------|-----------|--------------|-----------------|-------------------|
| 100    | 30%       | 48         | 99.35     | 51.57     | 1         | 171       | 99.5         | 0.3333          | 0.5692            |
| 200    | 20%       | 48         | 94.04     | 54.27     | 1         | 171       | 92           | 0.3333          | 0.5653            |
| 300    | 10%       | 47         | 90.60     | 56.45     | 1         | 171       | 97           | 0.3404          | 0.5582            |

### 1.2 Overlap-Inapplicable Strategies — Chunk Properties (Constant Across All Configs)

These strategies do not use the overlap parameter. Their chunk properties and scores are identical across all three configurations.

| Strategy   | Chunk Size | Overlap % | Num Chunks | Avg Words |
|------------|------------|-----------|------------|-----------|
| header     | var        | 0%        | 32         | 123.47    |
| semantic   | dynamic    | dynamic   | 10         | 407.10    |
| sentence   | ~3 sent    | 0%        | 55         | 74.07     |
| paragraph  | var        | 0%        | 56         | 69.55     |
| fixed_size | 1000       | 0%        | 47         | 86.64     |

---

## 2. Composite Score Comparison — All 28 Combinations

The composite score is a weighted aggregate of all retrieval and generation metrics.

### 2.1 Overlap-Sensitive Strategies (scores change across configs)

| Strategy    | Technique      | Composite @100 (30%) | Composite @200 (20%) | Composite @300 (10%) | Delta (100 to 300) |
|-------------|----------------|----------------------|----------------------|----------------------|--------------------|
| overlapping | hybrid         | 0.6495               | 0.6324               | 0.6153               | -0.0342            |
| recursive   | hybrid         | 0.6494               | 0.6425               | 0.6329               | -0.0165            |
| overlapping | dense_rerank   | 0.6449               | 0.6228               | 0.6120               | -0.0329            |
| overlapping | hybrid_rerank  | 0.6449               | 0.6228               | 0.6120               | -0.0329            |
| recursive   | dense_rerank   | 0.6406               | 0.6357               | 0.6122               | -0.0284            |
| recursive   | hybrid_rerank  | 0.6406               | 0.6357               | 0.6122               | -0.0284            |
| recursive   | dense          | 0.6175               | 0.6124               | 0.5908               | -0.0267            |
| overlapping | dense          | 0.6103               | 0.5945               | 0.5799               | -0.0304            |

### 2.2 Overlap-Immune Strategies (scores identical across all configs)

| Strategy   | Technique      | Composite (all configs) |
|------------|----------------|-------------------------|
| header     | hybrid         | 0.6343                  |
| semantic   | hybrid         | 0.6101                  |
| header     | dense          | 0.6101                  |
| fixed_size | hybrid         | 0.6079                  |
| paragraph  | hybrid         | 0.6031                  |
| sentence   | hybrid         | 0.6018                  |
| semantic   | dense          | 0.5986                  |
| sentence   | dense          | 0.5980                  |
| paragraph  | dense_rerank   | 0.5795                  |
| paragraph  | hybrid_rerank  | 0.5795                  |
| sentence   | dense_rerank   | 0.5840                  |
| sentence   | hybrid_rerank  | 0.5840                  |
| fixed_size | dense_rerank   | 0.5730                  |
| fixed_size | hybrid_rerank  | 0.5730                  |
| header     | dense_rerank   | 0.5726                  |
| header     | hybrid_rerank  | 0.5726                  |
| paragraph  | dense          | 0.5726                  |
| fixed_size | dense          | 0.5715                  |
| semantic   | dense_rerank   | 0.5018                  |
| semantic   | hybrid_rerank  | 0.5018                  |

---

## 3. Detailed Metric Breakdown — Overlap-Sensitive Strategies

### 3.1 recursive

#### recursive + dense

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.7652     | 0.7652     | 0.7348     | -0.0304             |
| Precision@K       | 0.2455     | 0.2364     | 0.2091     | -0.0364             |
| nDCG@K            | 0.4069     | 0.3932     | 0.3647     | -0.0422             |
| Recall@K          | 0.7121     | 0.7121     | 0.6939     | -0.0182             |
| Context Relevance | 0.6033     | 0.6028     | 0.6009     | -0.0024             |
| Faithfulness      | 0.6692     | 0.6697     | 0.6701     | +0.0009             |
| Answer Correct.   | 0.5020     | 0.5023     | 0.5008     | -0.0012             |
| Redundancy        | 0.6771     | 0.6784     | 0.6800     | +0.0029             |
| Token Cost        | 856        | 843        | 851        | -5                  |
| Latency (ms)      | 6.50       | 6.69       | 5.53       | -0.97               |
| **Composite**     | **0.6175** | **0.6124** | **0.5908** | **-0.0267**         |

#### recursive + hybrid

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.8106     | 0.8030     | 0.8106     | 0.0000              |
| Precision@K       | 0.2818     | 0.2727     | 0.2364     | -0.0454             |
| nDCG@K            | 0.4617     | 0.4422     | 0.4161     | -0.0456             |
| Recall@K          | 0.7159     | 0.7273     | 0.7273     | +0.0114             |
| Context Relevance | 0.5900     | 0.5904     | 0.5881     | -0.0019             |
| Faithfulness      | 0.6563     | 0.6576     | 0.6575     | +0.0012             |
| Answer Correct.   | 0.5009     | 0.5038     | 0.5025     | +0.0016             |
| Redundancy        | 0.6557     | 0.6543     | 0.6541     | -0.0016             |
| Token Cost        | 849        | 839        | 816        | -33                 |
| Latency (ms)      | 12.71      | 12.74      | 10.69      | -2.02               |
| **Composite**     | **0.6494** | **0.6425** | **0.6329** | **-0.0165**         |

#### recursive + dense_rerank

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.8045     | 0.8045     | 0.7742     | -0.0303             |
| Precision@K       | 0.2636     | 0.2545     | 0.2273     | -0.0363             |
| nDCG@K            | 0.4434     | 0.4300     | 0.3951     | -0.0483             |
| Recall@K          | 0.7159     | 0.7159     | 0.6977     | -0.0182             |
| Context Relevance | 0.5882     | 0.5845     | 0.5808     | -0.0074             |
| Faithfulness      | 0.6564     | 0.6528     | 0.6519     | -0.0045             |
| Answer Correct.   | 0.5021     | 0.5036     | 0.5016     | -0.0005             |
| Redundancy        | 0.6523     | 0.6427     | 0.6408     | -0.0115             |
| Token Cost        | 831        | 803        | 769        | -62                 |
| Latency (ms)      | 1789.81    | 1519.62    | 1420.43    | -369.38             |
| **Composite**     | **0.6406** | **0.6357** | **0.6122** | **-0.0284**         |

#### recursive + hybrid_rerank

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.8045     | 0.8045     | 0.7742     | -0.0303             |
| Precision@K       | 0.2636     | 0.2545     | 0.2273     | -0.0363             |
| nDCG@K            | 0.4434     | 0.4300     | 0.3951     | -0.0483             |
| Recall@K          | 0.7159     | 0.7159     | 0.6977     | -0.0182             |
| Context Relevance | 0.5882     | 0.5845     | 0.5808     | -0.0074             |
| Faithfulness      | 0.6564     | 0.6528     | 0.6519     | -0.0045             |
| Answer Correct.   | 0.5021     | 0.5036     | 0.5016     | -0.0005             |
| Redundancy        | 0.6523     | 0.6427     | 0.6408     | -0.0115             |
| Token Cost        | 831        | 803        | 769        | -62                 |
| Latency (ms)      | 1632.87    | 1463.06    | 1364.50    | -268.37             |
| **Composite**     | **0.6406** | **0.6357** | **0.6122** | **-0.0284**         |

### 3.2 overlapping

#### overlapping + dense

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8182     | 0.8182     | 0.8182     | 0.0000              |
| MRR               | 0.7424     | 0.7424     | 0.7197     | -0.0227             |
| Precision@K       | 0.2727     | 0.2364     | 0.2182     | -0.0545             |
| nDCG@K            | 0.4328     | 0.4012     | 0.3774     | -0.0554             |
| Recall@K          | 0.6932     | 0.6750     | 0.6750     | -0.0182             |
| Context Relevance | 0.6026     | 0.6032     | 0.6008     | -0.0018             |
| Faithfulness      | 0.6684     | 0.6696     | 0.6669     | -0.0015             |
| Answer Correct.   | 0.5009     | 0.5007     | 0.5012     | +0.0003             |
| Redundancy        | 0.6781     | 0.6803     | 0.6787     | +0.0006             |
| Token Cost        | 902        | 884        | 890        | -12                 |
| Latency (ms)      | 6.37       | 5.39       | 5.58       | -0.79               |
| **Composite**     | **0.6103** | **0.5945** | **0.5799** | **-0.0304**         |

#### overlapping + hybrid

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.7879     | 0.7689     | 0.7652     | -0.0227             |
| Precision@K       | 0.3000     | 0.2727     | 0.2364     | -0.0636             |
| nDCG@K            | 0.4746     | 0.4369     | 0.4007     | -0.0739             |
| Recall@K          | 0.7159     | 0.7273     | 0.7159     | 0.0000              |
| Context Relevance | 0.5891     | 0.5908     | 0.5880     | -0.0011             |
| Faithfulness      | 0.6552     | 0.6555     | 0.6568     | +0.0016             |
| Answer Correct.   | 0.4990     | 0.5011     | 0.4988     | -0.0002             |
| Redundancy        | 0.6586     | 0.6569     | 0.6596     | +0.0010             |
| Token Cost        | 870        | 856        | 829        | -41                 |
| Latency (ms)      | 12.78      | 10.44      | 11.36      | -1.42               |
| **Composite**     | **0.6495** | **0.6324** | **0.6153** | **-0.0342**         |

#### overlapping + dense_rerank

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.8045     | 0.7818     | 0.7742     | -0.0303             |
| Precision@K       | 0.2727     | 0.2455     | 0.2273     | -0.0454             |
| nDCG@K            | 0.4546     | 0.4180     | 0.3945     | -0.0601             |
| Recall@K          | 0.7159     | 0.6977     | 0.6977     | -0.0182             |
| Context Relevance | 0.5872     | 0.5826     | 0.5807     | -0.0065             |
| Faithfulness      | 0.6543     | 0.6502     | 0.6484     | -0.0059             |
| Answer Correct.   | 0.4993     | 0.4989     | 0.4985     | -0.0008             |
| Redundancy        | 0.6569     | 0.6465     | 0.6429     | -0.0140             |
| Token Cost        | 854        | 819        | 807        | -47                 |
| Latency (ms)      | 1793.23    | 1311.88    | 1468.58    | -324.65             |
| **Composite**     | **0.6449** | **0.6228** | **0.6120** | **-0.0329**         |

#### overlapping + hybrid_rerank

| Metric            | @100 (30%) | @200 (20%) | @300 (10%) | Delta (100 to 300) |
|-------------------|------------|------------|------------|---------------------|
| Hit@K             | 0.8636     | 0.8636     | 0.8636     | 0.0000              |
| MRR               | 0.8045     | 0.7818     | 0.7742     | -0.0303             |
| Precision@K       | 0.2727     | 0.2455     | 0.2273     | -0.0454             |
| nDCG@K            | 0.4546     | 0.4180     | 0.3945     | -0.0601             |
| Recall@K          | 0.7159     | 0.6977     | 0.6977     | -0.0182             |
| Context Relevance | 0.5872     | 0.5826     | 0.5807     | -0.0065             |
| Faithfulness      | 0.6543     | 0.6502     | 0.6484     | -0.0059             |
| Answer Correct.   | 0.4993     | 0.4989     | 0.4985     | -0.0008             |
| Redundancy        | 0.6569     | 0.6465     | 0.6429     | -0.0140             |
| Token Cost        | 854        | 819        | 807        | -47                 |
| Latency (ms)      | 1682.71    | 1248.88    | 1372.86    | -309.85             |
| **Composite**     | **0.6449** | **0.6228** | **0.6120** | **-0.0329**         |

---

## 4. Metric-Level Impact Analysis

This section identifies which individual metrics are most and least sensitive to overlap changes, based on the observed data across all overlap-sensitive combinations.

### 4.1 Metrics Most Affected by Overlap Reduction

| Metric        | Max Observed Delta | Strategy + Technique with Largest Drop       |
|---------------|--------------------|-----------------------------------------------|
| nDCG@K        | -0.0739            | overlapping + hybrid (0.4746 to 0.4007)       |
| Precision@K   | -0.0636            | overlapping + hybrid (0.3000 to 0.2364)       |
| nDCG@K        | -0.0601            | overlapping + dense_rerank (0.4546 to 0.3945) |
| Precision@K   | -0.0545            | overlapping + dense (0.2727 to 0.2182)        |
| nDCG@K        | -0.0554            | overlapping + dense (0.4328 to 0.3774)        |
| nDCG@K        | -0.0483            | recursive + dense_rerank (0.4434 to 0.3951)   |
| Precision@K   | -0.0454            | recursive + hybrid (0.2818 to 0.2364)         |

### 4.2 Metrics Least Affected by Overlap Reduction

| Metric            | Typical Delta Range | Observation                                          |
|-------------------|---------------------|------------------------------------------------------|
| Hit@K             | 0.0000              | Completely unchanged across all configs and combos    |
| Answer Correct.   | -0.0012 to +0.0016  | Negligible variation                                  |
| Context Relevance | -0.0074 to +0.0006  | Very low sensitivity                                  |
| Faithfulness      | -0.0059 to +0.0016  | Very low sensitivity                                  |
| Redundancy        | -0.0140 to +0.0029  | Low sensitivity, slight improvement at lower overlap  |

---

## 5. Strategy Sensitivity Ranking

Ranked by maximum composite score delta from config 100 (30%) to config 300 (10%), across all retriever techniques.

| Rank | Strategy    | Most Affected Technique | Composite Delta | Least Affected Technique | Composite Delta |
|------|-------------|-------------------------|-----------------|--------------------------|-----------------|
| 1    | overlapping | hybrid                  | -0.0342         | dense                    | -0.0304         |
| 2    | recursive   | dense_rerank            | -0.0284         | hybrid                   | -0.0165         |
| 3    | header      | (all)                   | 0.0000          | (all)                    | 0.0000          |
| 3    | semantic    | (all)                   | 0.0000          | (all)                    | 0.0000          |
| 3    | sentence    | (all)                   | 0.0000          | (all)                    | 0.0000          |
| 3    | paragraph   | (all)                   | 0.0000          | (all)                    | 0.0000          |
| 3    | fixed_size  | (all)                   | 0.0000          | (all)                    | 0.0000          |

---

## 6. Overlap-Immune Strategies — Verification

All five strategies below produced identical scores across all three overlap configurations. This was verified at both the Experiment Matrix level (same chunk count, avg words, corpus metrics) and the Heatmap level (same composite and per-metric scores).

| Strategy   | Overlap Used | Chunks | Composite (best technique) | Confirmed Identical |
|------------|--------------|--------|----------------------------|---------------------|
| header     | 0%           | 32     | 0.6343 (hybrid)            | Yes                 |
| semantic   | dynamic      | 10     | 0.6101 (hybrid)            | Yes                 |
| sentence   | 0%           | 55     | 0.6018 (hybrid)            | Yes                 |
| paragraph  | 0%           | 56     | 0.6031 (hybrid)            | Yes                 |
| fixed_size | 0%           | 47     | 0.6079 (hybrid)            | Yes                 |

---

## 7. Global Leaderboard — Top 10 Across All Configs

Ranked by composite score. Only the best overlap config for each strategy+technique pair is shown.

| Rank | Strategy    | Technique     | Best Overlap Config | Composite |
|------|-------------|---------------|---------------------|-----------|
| 1    | overlapping | hybrid        | 100 (30%)           | 0.6495    |
| 2    | recursive   | hybrid        | 100 (30%)           | 0.6494    |
| 3    | overlapping | dense_rerank  | 100 (30%)           | 0.6449    |
| 4    | overlapping | hybrid_rerank | 100 (30%)           | 0.6449    |
| 5    | recursive   | dense_rerank  | 100 (30%)           | 0.6406    |
| 6    | recursive   | hybrid_rerank | 100 (30%)           | 0.6406    |
| 7    | header      | hybrid        | any (identical)     | 0.6343    |
| 8    | recursive   | dense         | 100 (30%)           | 0.6175    |
| 9    | overlapping | dense         | 100 (30%)           | 0.6103    |
| 10   | semantic    | hybrid        | any (identical)     | 0.6101    |

---

## 8. Observations

### 8.1 What Changes with Overlap

- **Precision@K** and **nDCG@K** are the two metrics most consistently degraded when overlap is reduced. In the worst case (overlapping + hybrid), Precision@K drops from 0.3000 to 0.2364 and nDCG@K drops from 0.4746 to 0.4007.
- **MRR** shows moderate sensitivity, dropping by up to 0.0303 for rerank-based techniques.
- **Recall@K** drops modestly (up to -0.0182) in some combinations but stays flat or slightly increases in others.
- **Hit@K** is completely invariant to overlap changes across all tested combinations.
- **Generation-quality metrics** (Context Relevance, Faithfulness, Answer Correctness) show negligible sensitivity to overlap changes, with deltas under 0.01 in all cases.

### 8.2 Strategy Behavior Under Overlap Reduction

- **overlapping** shows the largest composite drops across all four retriever techniques (range: -0.0304 to -0.0342). Its Precision@K and nDCG@K degrade more steeply than recursive under the same conditions.
- **recursive** shows smaller composite drops (range: -0.0165 to -0.0284). When paired with `hybrid`, it retains the most performance under overlap reduction (only -0.0165 composite delta).
- **recursive + dense_rerank** and **recursive + hybrid_rerank** produce identical scores at every overlap config, indicating the reranker produces the same ranking output regardless of the initial retrieval technique for this strategy.
- **overlapping + dense_rerank** and **overlapping + hybrid_rerank** also produce identical scores at every overlap config, showing the same reranker convergence behavior.

### 8.3 Chunk Property Changes

- Reducing overlap from 30% to 10% decreases average words per chunk (recursive: 96.4 to 87.3; overlapping: 99.35 to 90.6).
- Chunk count drops by 1 (48 to 47) at the lowest overlap setting for both strategies.
- Corpus redundancy decreases with lower overlap (recursive: 0.5676 to 0.5550; overlapping: 0.5692 to 0.5582), which is expected as less text is duplicated across chunk boundaries.

---

## 9. Data Completeness Notes

- All 28 strategy+technique combinations were evaluated under each of the 3 overlap configurations, totaling 84 experiments.
- All 22 golden dataset questions were used for every experiment.
- The 5 overlap-immune strategies (header, semantic, sentence, paragraph, fixed_size) were verified to produce byte-identical results across all 3 configs at both the Experiment Matrix and Heatmap sheet levels.
- dense_rerank and hybrid_rerank produce identical metric scores for the same strategy and overlap config in all observed cases, differing only in latency.
