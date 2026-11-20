# 1MS25CI117 – Champion–Challenger Model Comparison with Hugging Face and MLflow
**Student:** Shashikala · **USN:** 1MS25CI117 · **Section:** B · Skill Lab-1 (CISL-1)

## 1. Task and models
| Item | Value |
|---|---|
| Task | Binary sentiment classification (negative / positive) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`, Hugging Face Hub), human-labelled movie-review sentences |
| Subset | 200 sentences: 100 negative and 100 positive, seed 117 → `data/eval_inputs.csv` |
| Model A – current Champion | `distilbert-base-uncased-finetuned-sst-2-english` (~67M, table row 1) |
| Model B – Challenger | `siebert/sentiment-roberta-large-english` (~355M, table row 2) |
| Inference settings | CPU, batch size 1, max_length 128, label = argmax of softmax; model labels `NEGATIVE`/`POSITIVE` mapped to SST-2 ids 0/1 |

Pairing "1 vs 2" is one of the suggested classification pairings in the brief.

## 2. How to run
```bash
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
# in a second terminal:
python evaluate.py                       # runs A and B on the same 200 sentences, logs everything, makes the decision
python register_champion.py              # registers both versions, sets @champion / @challenger aliases
python predict_from_mlflow.py            # loads models:/1MS25CI117-sentiment-model@champion and classifies data/new_data.csv
python predict_from_mlflow.py challenger # stretch goal: same script, other alias, no code change
```
MLflow experiment: `1MS25CI117-hf-sentiment-champion-challenger` · Registered model: `1MS25CI117-sentiment-model`
(To use another tracking server, set `MLFLOW_TRACKING_URI`. No run IDs are hard-coded.)

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 (%) on the shared 200-sentence set. **Higher is better.** Macro-F1 weights both classes equally, so a model cannot win by favouring one label.
- **Margin:** B must beat A by **at least 1.5 macro-F1 points**. With 200 sentences, one sentence is worth about 0.5 points, so the Challenger has to get roughly 3 more sentences right before it counts; anything smaller could be 1–2 lucky sentences.
- **Guardrail:** B is rejected if its average latency is **more than 10×** A's. **Lower latency is better.** B is about 5× bigger, so some slowdown is expected and acceptable; 10× would be too slow for interactive CPU use.
- **Tie-breaker:** if |Δ macro-F1| < 1.5, keep the **faster** model.
- Secondary metrics (reported, not used for the decision): accuracy ↑, precision (positive) ↑, recall (positive) ↑, total inference time ↓.

## 4. Results
| Metric | A: DistilBERT-SST2 | B: RoBERTa-large | B − A |
|---|---:|---:|---:|
| Accuracy ↑ | 90.00 | 93.50 | +3.50 |
| **Macro-F1 ↑ (primary)** | **89.99** | **93.50** | **+3.51** |
| Precision (positive) ↑ | 87.74 | 92.23 | +4.50 |
| Recall (positive) ↑ | 93.00 | 95.00 | +2.00 |
| Avg latency (ms/sentence) ↓ | 49.48 | 282.45 | 5.71× |
| Total inference (s) ↓ | 9.90 | 56.49 | +46.59 |

Errors on the 200 sentences: A gets 20 wrong (13 false positives, 7 false negatives), B gets 13 wrong (8 false positives, 5 false negatives). The models disagree on 15 sentences (`outputs/disagreements.csv`): B is right on 11 of them and A on 4. Five are commented in `outputs/qualitative_examples.csv`.

Plots: `outputs/metric_comparison.png` (every metric, A vs B), `outputs/confusion_matrix_distilbert-sst2.png`, `outputs/confusion_matrix_roberta-large.png`, `outputs/confidence_distribution.png` (confidence of correct vs wrong predictions).

## 5. Decision – New Champion: Model B (`siebert/sentiment-roberta-large-english`)
Yes, the Challenger improves on the Champion. Its macro-F1 is 93.50 against 89.99, a gain of +3.51 points, which is more than double the 1.5-point margin fixed before the run. Its latency is 5.71× DistilBERT's (282 ms vs 49 ms per sentence), which is inside the 10× guardrail, so the rule promotes B. B now holds the `@champion` alias (version 2) and DistilBERT is kept as version 1 with `@challenger`.

Is the gain meaningful? It is 7 sentences out of 200, so it is not huge, but it is consistent: B is better on accuracy, macro-F1, precision and recall, and on the 15 sentences where the models disagree B is right 11 times to A's 4. The biggest gain is fewer false positives (8 vs 13). B handles sarcasm, comparisons and "but" sentences better, for example "I'll bet the video game is a lot more fun than the film". A different random 200 could move the gap by a point or two, so I would call B clearly better, but not by exactly 3.5 points.

Trade-offs: B is about 5× bigger (355M vs 67M parameters, ~1.4 GB vs ~270 MB on disk) and about 5.7× slower on CPU. For batch scoring or a low-traffic app that is fine. If the model had to run on a phone or answer thousands of requests per second, DistilBERT would still be the practical choice. Both models are almost always over 95% confident, even when wrong, so confidence alone cannot be used to catch errors.

## 6. Limitations and declaration
- Only 200 SST-2 sentences (movie reviews), so ±1–2 macro-F1 points is within noise. Results may differ on tweets, product reviews or Indian English.
- DistilBERT was fine-tuned on the SST-2 training set, and SiEBERT on a mix of English sentiment datasets, so both are tested on the kind of text they were trained for. The evaluation uses only the validation split, never training data.
- SST-2 has no neutral class. Neither model can say "neutral", so mixed sentences are forced into positive or negative.
- Latency was measured on a 2-core CPU with batch size 1 and varies between runs. The ratio is more reliable than the absolute ms.
- `predict_from_mlflow.py` never calls Hugging Face directly. It loads the model through the alias URI, and the pyfunc wrapper calls transformers inside MLflow. The version and HF model ID are read from the registry tags.
- New data (`data/new_data.csv`, 8 sentences) was written by me and is not from SST-2. Both models label the sarcastic new_004 ("Oh great, another surprise quiz…") as positive, which is wrong. Only the mixed review new_005 changes between the aliases: the new Champion says negative, DistilBERT says positive (`outputs/new_data_predictions*.csv`).
- AI assistant used: Hark (AI assistant) helped write and debug the scripts and README. I can explain every line.

## 7. Repository structure
```
data/        eval_inputs.csv, new_data.csv
outputs/     predictions_*.csv, metrics_summary*.json, model_comparison.csv, disagreements.csv, qualitative_examples.csv,
             metric_comparison.png, confusion_matrix_*.png, confidence_distribution.png,
             new_data_predictions*.csv, model_runs.json, *_terminal.txt
screenshots/ MLflow runs, compare view (params + metrics), summary run, registry aliases, predict_from_mlflow.py and register terminal output
evaluate.py, hf_wrapper.py, register_champion.py, predict_from_mlflow.py, requirements.txt, .gitignore
```
