# HindiNLPToolKit

A Python library and web app for studying **word order in Hindi**. It takes
Hindi sentences, keeps the ones whose preverbal phrases can be reordered,
generates every grammatical reordering ("variants"), scores the original and
each variant with five predictors, and measures how often each predictor picks
the order a human actually wrote. For a single typed sentence it recommends the
most natural order.

The method follows Ranjan & van Schijndel (2024), *Does Dependency Locality
Predict Non-canonical Word Order in Hindi?* (CogSci 2024,
[arXiv:2405.07730](https://arxiv.org/abs/2405.07730)).

| Scorer | What it measures | Trained model |
|---|---|---|
| `dependency_length` | total dependency length of the tree | none |
| `information_status` | given-before-new ordering | none |
| `trigram` | Katz-backoff trigram surprisal (Hindi Wikipedia) | `trigram_katz/` |
| `adaptive_lstm_wiki50m` | adaptive LSTM surprisal (50.8 M Wikipedia tokens) | `base_model_wiki50m.pt` + `vocab.pkl` |
| `berkeley_pcfg` | whole-sentence PCFG surprisal (DS-PS grammar) | `hdtb_dsps_grammar` |

The web app also bundles the **Taru** tree and incremental-surprisal tool at
`/taru` (third-party, shipped as-is with its own grammar files).

---

## Repository map

```
stanza_parser/ filtering/ variants/ helpers/   the pipeline library
scoring/              the five scorers (plugins); scoring/models/ = trained models (not in git)
evaluation/           pairwise ranking accuracy: 10-fold CV logistic regression
webapp/               FastAPI web app (imports the library, never the reverse)
taru/                 bundled Taru tree / surprisal tool + Berkeley Parser jar
data/                 the two treebanks (see NOTICE.md)
training/             scripts that build every model + Terraform to run them on AWS
reproducibility/      rebuild the paper-comparison table from scratch
deploy/               Docker image -> one EC2 instance, with Terraform
tests/test_all.py     integration test suite
DOCS.md               full API reference          CONTRIBUTING.md   adding a scorer
```

---

## 1. Install (local)

Python **3.11**, and **Java 11+** on `PATH` for the PCFG scorer.

```bash
git clone https://github.com/<account>/<this-repo>.git
cd <this-repo>
python -m venv venv
source venv/bin/activate                 # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` pins every package to the versions the results were produced
with. The first `.txt` / typed-sentence parse downloads the Stanza Hindi model
(~1 GB) to `~/stanza_resources/`.

## 2. Get the models

The trained models are **not** in the repository. Build them with the scripts in
[`training/`](training/README.md), on your own machine or on one AWS GPU
instance:

```bash
bash training/run_all.sh            # local Linux / macOS / WSL; a GPU helps the LSTM
bash training/train_models.sh       # or on AWS: ~2 h, ~$1-2, results land in scoring/models/
```

Afterwards `scoring/models/` holds `trigram_katz/` (10 files),
`base_model_wiki50m.pt`, `vocab.pkl`, `hdtb_dsps_grammar` and the committed
`recommender_weights.json`. Without them the library still runs; the
model-based scorers return `NaN`.

## 3. Run

```bash
python tests/test_all.py                                     # full test suite (needs the models)
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000  # web app at http://localhost:8000
```

Library use, in short (details in [`DOCS.md`](DOCS.md)):

```python
from stanza_parser import load_input
from filtering import filter_sentences
from variants import generate_variants
from scoring import apply_scorers, build_corpus_context
from evaluation import evaluate_pairs

sents = load_input("data/hi_hdtb-ud-train.conllu")
passed, rejected_df, passed_df = filter_sentences(sents, allowed_root_pos=["VERB", "AUX"])
pairs = generate_variants(passed)
pairs = apply_scorers(pairs, ["dependency_length", "trigram"],
                      context={"corpus": build_corpus_context(sents),
                               "passed": passed, "scheme": "ud"})
print(evaluate_pairs(pairs)["predictors"])
```

## 4. Reproduce the results

[`reproducibility/`](reproducibility/README.md) rebuilds the full feature table
from the HDTB treebank (~65 min) and fits the paper's logistic regressions:

```bash
python reproducibility/build_features.py
python reproducibility/evaluate.py      # -> reproducibility/results/*.csv
```

The committed `reproducibility/results/*.csv` are what these two commands
produce with the shipped models.

---

## 5. Deploy

### 5a. Locally, with Docker

Put the models in `scoring/models/` first — they are copied into the image.

```bash
docker build -t hindi-nlp-toolkit .
docker run --rm -p 8001:8001 hindi-nlp-toolkit
```

Open http://localhost:8001/ (pipeline), `/taru` (tree tool) and
`/how-it-works`. The image installs Java, compiles Taru's C++ engine and bakes
in the Stanza model; the first build takes 10-25 minutes.

### 5b. AWS credentials (your own account)

Nothing in this repository is tied to an AWS account: the scripts use whatever
credentials your AWS CLI is configured with. Create a dedicated IAM user rather
than using root keys:

1. AWS console → **IAM → Policies → Create policy → JSON**: paste
   [`deploy/iam-policy.json`](deploy/iam-policy.json) and name it
   `hindi-nlp-deployer`. It allows only what this repo's Terraform creates:
   EC2 instances and security groups, `hindi-nlp-*` S3 buckets and IAM roles,
   and SSM shell sessions.
2. **IAM → Users → Create user** (e.g. `hindi-nlp-deployer`) and attach that policy.
3. Open the user → **Security credentials → Create access key → Command Line
   Interface**, and copy the key id and secret (shown only once).
4. Install the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   and [Terraform ≥ 1.5](https://developer.hashicorp.com/terraform/install), then:

   ```bash
   aws configure --profile hindi-nlp       # key, secret, region us-east-1, output json
   export AWS_PROFILE=hindi-nlp            # PowerShell: $env:AWS_PROFILE = "hindi-nlp"
   aws sts get-caller-identity             # prints your account id if it worked
   ```

The keys stay in `~/.aws/credentials` on your machine. Terraform keeps its state
(the ids of what it created) in local `terraform.tfstate` files, which are
git-ignored. Never commit either.

### 5c. Deploy to AWS (one EC2 instance)

```bash
bash deploy/deploy.sh        # Windows: .\deploy\deploy.ps1
```

This builds the image, gzips it and runs `terraform apply`, which shows the plan
and asks before creating anything: one S3 bucket for the image, an IAM role, a
security group open on port 80 only, and a `t3.xlarge` instance that loads and
runs the container. It prints `http://<public-ip>/`; the app answers about
5 minutes later.

| Action | Command | Cost (us-east-1) |
|---|---|---|
| running | — | ~$0.17/hour |
| pause between sessions | `deploy/stop.sh` (`stop.ps1`) | ~$2.40/month (disk only) |
| resume — prints the new IP | `deploy/start.sh` (`start.ps1`) | — |
| remove everything | `deploy/destroy.sh` (`destroy.ps1`) | ~$0 |

Shell access without SSH: `aws ssm start-session --target <instance-id>`, then
`docker logs hindi-nlp`. More in [`deploy/README.md`](deploy/README.md).

---

## Licences

The toolkit's own code is published without a licence grant (all rights
reserved). Bundled data and third-party tools carry their own terms — see
[`NOTICE.md`](NOTICE.md). In particular UD Hindi-HDTB is CC BY-NC-SA 4.0
(non-commercial use).
