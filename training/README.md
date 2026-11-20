# training/ — build every model the scorers use

Three scorers need trained models. Everything here is plain Python / shell that
you can read top to bottom; the Terraform stack only rents a machine to run it.

```
training/
  wiki/       download.py  extract.py  sentences.py  build_vocab.py   Hindi Wikipedia -> sentences + LSTM vocabulary
  trigram/    train_trigram.py                                         Katz-backoff trigram
  lstm/       train_lstm.py  lm.py  eval_perplexity.py                 2-layer LSTM language model
  pcfg/       generate_dsps_mrg.py  ds_ps_ud/  train_grammar.sh        DS-PS conversion + Berkeley PCFG
  run_all.sh                 runs the whole chain on one machine
  train_models.sh / .ps1     runs run_all.sh on one AWS GPU instance via terraform/, then fetches the models
  terraform/                 that AWS stack (S3 bucket, IAM role, SG, one self-terminating EC2 instance)
  requirements-train.txt     extra pip packages (mwxml, mwparserfromhell)
  weights.md                 how the recommender weights were fitted
```

## What gets built

| Step | Script | Input | Output (`scoring/models/`) | Time | Details |
|---|---|---|---|---|---|
| wiki | `wiki/download.py` → `extract.py` → `sentences.py` | Hindi Wikipedia dump | `training/work/wiki_sentences.txt` | ~45 min | [wiki/README.md](wiki/README.md) |
| vocab | `wiki/build_vocab.py` | wiki_sentences.txt | `vocab.pkl` | ~3 min | [wiki/README.md](wiki/README.md) |
| trigram | `trigram/train_trigram.py` | wiki_sentences.txt | `trigram_katz/` | ~12 min | [trigram/README.md](trigram/README.md) |
| lstm | `lstm/train_lstm.py` | wiki_sentences.txt + vocab.pkl | `base_model_wiki50m.pt` | 25 min (GPU) | [lstm/README.md](lstm/README.md) |
| pcfg | `pcfg/generate_dsps_mrg.py` → `train_grammar.sh` | `data/hi_hdtb-ud-train.conllu` | `hdtb_dsps_grammar` | ~15 min | [pcfg/README.md](pcfg/README.md) |

`dependency_length` and `information_status` need no model.
`recommender_weights.json` is committed; see [weights.md](weights.md).
The bundled Taru tool's grammar and synproc model are shipped as-is (no
training script).

## Option A — on your own machine

Linux, macOS or WSL, Python 3.11 with the toolkit installed (README §1), Java
11+, ~5 GB free disk, 8 GB RAM. An NVIDIA GPU makes the LSTM step meaningful:
the step trains for a fixed 25 minutes, and a CPU gets through far less data
in that time.

```bash
pip install -r requirements.txt -r training/requirements-train.txt -c constraints.txt
bash training/run_all.sh                       # all steps
STEPS="trigram pcfg" bash training/run_all.sh  # some steps (later steps reuse earlier outputs)
WIKI_DATE=20260201 bash training/run_all.sh    # pin a Wikipedia dump
LSTM_ARGS="--minutes 60" bash training/run_all.sh
```

Each script also runs on its own; every one has `--help` and sensible defaults
(inputs from `training/work/` and `data/`, outputs into `scoring/models/`).

## Option B — on AWS with one command

Needs Terraform ≥ 1.5, AWS CLI v2, git, and AWS credentials for **your own**
account (README §5b — the same IAM policy covers this stack).

```bash
bash training/train_models.sh          # Windows: .\training\train_models.ps1
```

What happens:

1. The repository's tracked files are packed into `training/terraform/build/code.tar.gz`.
2. `terraform apply` (it shows the plan and asks for `yes`) creates a private S3
   bucket, uploads the tarball, and starts one `g4dn.xlarge` (NVIDIA T4, the GPU
   the shipped LSTM was trained on) on the AWS Deep Learning AMI, with no inbound
   ports.
3. The instance installs Java and the training packages, runs `run_all.sh`,
   uploads `models.tar.gz` and its log to the bucket, writes a `DONE` marker and
   **terminates itself**. A dead-man timer (`dead_man_minutes`, default 240)
   terminates it anyway if anything hangs.
4. The script polls the log every 2 minutes, downloads the models into
   `scoring/models/` (moving any existing ones to `scoring/models.bak-<time>`),
   and runs `terraform destroy` (asks again).

Cost: ~2 hours of `g4dn.xlarge` ≈ **$1-2** on demand in us-east-1, plus pennies
of S3. Options are Terraform variables, set as environment variables:

```bash
TF_VAR_instance_type=g5.xlarge  TF_VAR_steps="lstm"  TF_VAR_wiki_date=20260201 \
TF_VAR_lstm_args="--minutes 60"  TF_VAR_region=us-west-2  bash training/train_models.sh
```

⚠ **GPU quota.** New AWS accounts often have a quota of **0** for GPU
instances, and `apply` then fails with `VcpuLimitExceeded`. Request an increase
first: console → **Service Quotas → Amazon EC2 → "Running On-Demand G and VT
instances"** → request **4** (or more) vCPUs. Approval can take a day.
Alternatively set `TF_VAR_instance_type=c7i.4xlarge` (CPU only; the LSTM then
sees fewer training steps in its time budget).

Watch a run by hand: `aws s3 cp s3://<bucket>/logs/train.log -` or
`aws ssm start-session --target <instance-id>` then `tail -f /var/log/hnt-train.log`.

## How exact is a retrain?

| Model | Deterministic? | Notes |
|---|---|---|
| DS-PS trees (`hdtb_train_dsps.mrg`) | yes | identical to the original (MD5 `5b558660…`, 13,282 trees) |
| `vocab.pkl`, `trigram_katz/` | yes, given the same Wikipedia dump | seeded split (42); the shipped models used the dump fetched 2026-02-19 (see wiki/README.md) |
| `hdtb_dsps_grammar` | yes on the same JVM | Berkeley's trainer uses a fixed default seed |
| `base_model_wiki50m.pt` | **no** | stops on a wall-clock budget, so GPU speed changes how many steps it gets; GPU kernels are not bit-reproducible either |

So a retrain reproduces the trigram and PCFG models and gives an LSTM of the
same design and similar quality, not identical weights. The shipped LSTM reached
held-out perplexity 79.65; expect a similar value on a T4 and somewhat better on
a faster GPU. Accuracy figures in `reproducibility/results/` were produced with
the shipped models.
