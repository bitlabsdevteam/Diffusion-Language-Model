"""Offline tests for fine-tuning/ft_train.py + ft_lora.py on a tiny model (CPU, seconds):
python fine-tuning/tests/test_ft_train.py"""
import contextlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
from _check import check, exit_with  # noqa: E402

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import ft_data as D  # noqa: E402
import ft_lora as L  # noqa: E402
import ft_train as T  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM, load_model  # noqa: E402

TOK_PATH = _path.DLM_ROOT / "tokenizer" / "tokenizer.json"
tok = Tokenizer.from_file(str(TOK_PATH))


def raw_record(i):
    w1, w2 = f"cat{i % 5}", f"dog{i % 7}"
    fields = [f"Words: {w1}, {w2}", f"Summary: A {w1} met a {w2}."]
    body = f"The {w1} met the {w2}. They played. The {w1} said hello."
    return "\n".join(fields) + "\nStory: \n" + body + "\n"


def run(argv, tmp, quiet=True):
    a = T.build_parser().parse_args(argv)
    a.train_examples = D.SftCache(a.cache, "train").manifest["splits"]["train"]["examples"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        res = T.run_sft(a, Path(a.out), tok)
    return res, buf.getvalue()


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    raw = d / "raw.txt"
    raw.write_text("<|endoftext|>\n".join(raw_record(i) for i in range(400)), encoding="utf-8")
    D.build_cache(raw, d / "cache", "train", TOK_PATH)
    D.build_cache(raw, d / "cache", "valid", TOK_PATH, limit=100)
    torch.manual_seed(0)
    cfg = DLMConfig.from_preset("tiny", vocab_size=8192, seq_len=256, pad_id=0, mask_id=4)
    base = DiffusionTransformerLM(cfg)
    base.save_pretrained(d / "pre")
    (d / "pre" / "run_info.json").write_text(json.dumps({"recipe": {"schedule": "cosine"}}))
    check(T.infer_schedule(d / "pre") == "cosine" and T.infer_schedule(d) == "linear", "infer_schedule reads run_info.json")

    common = ["--init-from", str(d / "pre"), "--cache", str(d / "cache"), "--pretrain-cache", str(d / "none"),
              "--tokenizer-dir", str(_path.DLM_ROOT / "tokenizer"), "--seq-len", "256", "--batch-size", "4",
              "--lr", "3e-3", "--warmup", "5", "--eval-every", "30", "--ckpt-every", "30", "--log-every", "10",
              "--print-every", "10", "--gate-uncond-max", "0", "--val-examples", "60", "--device", "cpu", "--precision", "fp32", "--no-such" if False else "--seed", "1"]

    # ---- straight run, 60 steps
    res, log = run(common + ["--steps", "60", "--out", str(d / "full")], d)
    h = res["history"]
    base_e = next(x for x in h if x.get("base"))
    evals = [x for x in h if "val" in x and not x.get("base")]
    check(res["final"] and len(evals) == 2, f"two evaluations after the base one ({len(evals)})")
    check(evals[-1]["val"] < base_e["val"] - 0.3, f"SFT validation bound falls ({base_e['val']:.2f} -> {evals[-1]['val']:.2f})")
    check(all(x["loss"] == x["loss"] for x in h if "loss" in x), "no NaN losses")
    fin = d / "full" / "final"
    check((fin / "model.pt").is_file() and (fin / "config.json").is_file() and (fin / "run_info.json").is_file() and (fin / "tokenizer" / "tokenizer.json").is_file(), "final/ complete")
    check((d / "full" / "best" / "model.pt").is_file(), "best/ written")
    m = load_model(fin)
    check(m.config.to_dict() == cfg.to_dict(), "final model loads with the same config")
    info = json.loads((fin / "run_info.json").read_text())
    check(info["task"] == "sft" and info["recipe"]["schedule"] == "cosine" and info["base_eval"]["base"], "run_info has task, auto schedule, base eval")
    check("EVAL step" in log and "step 0 (base model)" in log, "log has the base and EVAL lines")
    check((d / "full" / "history.json").is_file(), "history.json written")

    # ---- resume is bit-exact on CPU: 30 + resume 30 == 60 straight
    res1, _ = run(common + ["--steps", "60", "--stop-at", "30", "--out", str(d / "split")], d)
    check(not res1["final"] and (d / "split" / "resume.pt").is_file(), "stop-at writes resume.pt")
    res2, log2 = run(common + ["--steps", "60", "--resume", "--out", str(d / "split")], d)
    check("resumed from step 30" in log2, "resumed from step 30")
    a, b = torch.load(d / "full" / "final" / "model.pt"), torch.load(d / "split" / "final" / "model.pt")
    check(all(torch.equal(a[k], b[k]) for k in a), "resumed run == uninterrupted run (bit-exact weights)")
    # a changed recipe refuses to resume
    try:
        run(common + ["--steps", "60", "--lr", "1e-3", "--resume", "--out", str(d / "split")], d)
        check(False, "changed recipe must refuse to resume")
    except SystemExit as e:
        check("Refusing to resume" in str(e), "changed recipe refuses to resume")

    # ---- step-0 gate: a base model near the unigram baseline must abort (needs the real pre-training cache; skipped without it)
    pre = _path.DLM_ROOT / "data" / "tokens"
    if (pre / "tokens_manifest.json").is_file():
        gate_args = [x for x in common if x not in ("--gate-uncond-max", "0")]
        gate_args = gate_args[:gate_args.index("--pretrain-cache") + 1] + [str(pre)] + gate_args[gate_args.index("--pretrain-cache") + 2:]
        try:
            run(gate_args + ["--steps", "10", "--out", str(d / "gated")], d)
            check(False, "an untrained base model must trip the step-0 gate")
        except T.SftError as e:
            check("STEP 0 GATE" in str(e), "untrained base model trips the step-0 gate")
        check(not (d / "gated" / "final").exists(), "no final/ after a gate abort")
    else:
        print("  (no pre-training token cache: gate test skipped)")

    # ---- LoRA arm
    resl, logl = run(common + ["--steps", "60", "--lora-r", "4", "--out", str(d / "lora")], d)
    check("LoRA r=4" in logl, "LoRA announced")
    ml = load_model(d / "lora" / "final")
    check(not L.has_lora(ml), "exported LoRA checkpoint is a plain merged model")
    p0 = load_model(d / "pre")
    check(torch.equal(ml.tok_emb.weight, p0.tok_emb.weight), "embedding / lm_head untouched by LoRA")
    check(ml.lm_head.weight is ml.tok_emb.weight, "tie preserved after merge")
    check(not torch.equal(ml.blocks[0].attn.qkv.weight, p0.blocks[0].attn.qkv.weight), "attention weights changed by LoRA")
    check(torch.equal(ml.blocks[0].norm1.weight, p0.blocks[0].norm1.weight), "norm gains frozen under LoRA")
    el = [x for x in resl["history"] if "val" in x and not x.get("base")]
    check(el[-1]["val"] < base_e["val"], "LoRA also reduces the validation bound")

# ---- LoRA unit checks
torch.manual_seed(3)
m = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=300, seq_len=64, pad_id=0, mask_id=4))
x = torch.randint(5, 300, (2, 20))
y0 = m(x).detach()
n_total = m.num_parameters()
n_tr = L.apply_lora(m, r=4)
check(torch.allclose(m(x), y0, atol=1e-6), "zero-init B: LoRA model == base model at the start")
exp = 2 * 4 * ((64 + 192) + (64 + 64) + (64 + 176) * 3) * 2  # r*(in+out) per target; tiny: d=64, d_ff=176, 2 layers
check(n_tr == 4 * ((64 + 192) + (64 + 64) + 3 * (64 + 176)) * 2, f"trainable count {n_tr}")
check(n_tr < 0.5 * n_total, "adapters are a fraction of the model")
names = {n for n, p in m.named_parameters() if p.requires_grad}
check(all(n.endswith(".A") or n.endswith(".B") for n in names), "only A/B train")
check(not any("lm_head" in n or "tok_emb" in n or "norm" in n for n in names), "embedding, head and norms frozen")
for mod in m.modules():
    if isinstance(mod, L.LoRALinear):
        torch.nn.init.normal_(mod.B, std=0.05)
y1 = m(x).detach()
check(not torch.allclose(y1, y0, atol=1e-4), "adapters change the output once B != 0")
merged = L.merge_lora(m)
check(not L.has_lora(merged) and torch.allclose(merged(x).detach(), y1, atol=1e-4), "merged model reproduces the LoRA model")
check(L.has_lora(m), "merge_lora leaves the original untouched")
check(merged.lm_head.weight is merged.tok_emb.weight, "tie kept")
try:
    L.apply_lora(m, r=0)
    check(False, "rank 0 must be rejected")
except ValueError:
    check(True)

exit_with("test_ft_train")
