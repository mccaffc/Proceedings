#!/usr/bin/env python3
"""Make the engraved artwork for papers that lack it.

For each ICMI-*.md paper without assets/art/band/<key>.jpg or assets/art/tile/<key>.jpg (or for the keys given
with --force), this:
  1. picks a subject: assets/art/subjects.json (key -> subject), else Claude proposes one from the title and
     abstract, and the choice is written back to subjects.json so a person can edit it and rerun with --force;
  2. generates a grey tonal study (OpenRouter, Nano Banana) into assets/art/src/<key>.png;
  3. engraves it (scripts/art/engrave.py) into the band and tile JPEGs and their field PNGs.

  python3 scripts/art/make-art.py                  # every paper that lacks art
  python3 scripts/art/make-art.py --force 29 B     # redo these, generating a new study from subjects.json
  python3 scripts/art/make-art.py --dry-run        # list what would be done
  python3 scripts/art/make-art.py --out DIR        # write DIR/art/... instead of assets/art/... (testing)

Keys: OPENROUTER_API_KEY (locally OPENROUTER_FAI_API_KEY also works) and ANTHROPIC_API_KEY. If a key is missing,
the script says so and exits 0. Needs numpy, scipy and Pillow.
"""
import argparse, base64, io, json, os, re, shutil, ssl, subprocess, sys, tempfile, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGRAVE = Path(__file__).with_name("engrave.py")
ART = REPO / "assets" / "art"

STUDY_MODEL = "google/gemini-nano-banana-2.1"
SUBJECT_MODEL = os.environ.get("ICMI_ART_MODEL", "claude-sonnet-5-5")
STUDY_PROMPT = (
    "Smooth monochrome grayscale tonal study of a single object, like a carved plaster or matte clay sculpture lit "
    "softly from the upper left, mid-to-dark grey forms with strong contrast and deep shadows, on a pure flat white "
    "background. Shading fully smooth with no texture, no lines, no hatching, no grain. One large, bold, legible "
    "form with a clear silhouette, centred, filling about two thirds of the height, with empty white space around "
    "it. No text, no letters, no border, no frame, no floor line. Subject: ")
JPEG_QUALITY = 80
BAND_SIZE = (2160, 780)      # the site's band JPEGs: the 2880x1040 engraving at 75%


def log(*a):
    print(*a, flush=True)


def ssl_ctx():
    for cafile in ("/etc/ssl/cert.pem",):
        if not os.environ.get("SSL_CERT_FILE") and os.path.exists(cafile) and sys.platform == "darwin":
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def post_json(url, headers, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, context=ssl_ctx(), timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s from %s: %s" % (e.code, url.split("/")[2], e.read()[:300].decode("utf-8", "replace")))


# ---------------------------------------------------------------- papers
def papers():
    """[{key, seed, file, title, abstract}] for every ICMI-*.md, with keys and seeds as render-site.py derives them."""
    out = []
    for md in sorted(REPO.glob("ICMI-*.md")):
        m = re.match(r"ICMI-([0-9A-Za-z]+)-", md.name)
        if not m:
            continue
        num = m.group(1)
        is_num = num.isdigit()
        key = "%02d" % int(num) if is_num else num
        seed = (336 + 7 * (int(num) % 10)) if is_num else 455 + 7 * (ord(num[:1].upper()) - ord("A"))
        text = md.read_text(encoding="utf-8")
        t = re.search(r"^#\s+(.+)$", text, re.M)
        a = re.search(r"\*\*Abstract\.?\*\*\.?\s*(.+?)(?:\n\s*\n|\Z)", text, re.S)
        out.append({"key": key, "seed": seed, "file": md.name,
                    "title": re.sub(r"[*_`]", "", t.group(1)).strip() if t else md.stem,
                    "abstract": re.sub(r"\s+", " ", a.group(1)).strip()[:2500] if a else ""})
    return out


def has_art(key):
    return all((ART / k / (key + ".jpg")).exists() and (ART / "field" / ("%s-%s.png" % (k, key))).exists()
               for k in ("band", "tile"))


# ---------------------------------------------------------------- subject
def propose_subject(p):
    key = os.environ.get("ANTHROPIC_API_KEY")
    prompt = (
        "A research publication series from a Christian institute (machine intelligence and virtue) gives every "
        "paper an emblematic engraved illustration of one object. Propose the single object for this paper.\n\n"
        "Rules: one concrete, physical, emblematic object that a sculptor could carve from plaster. Biblical, "
        "liturgical or everyday iconography is fine (a lamp, a bowl, a trumpet, a sheaf, a key, a ladder, a "
        "millstone). No people or faces, no text, no crosses (the cross belongs to the logo), no scenes, no "
        "animals in action. Choose something that echoes the paper's central idea, not a generic religious symbol.\n\n"
        "Reply with only the subject, as a short noun phrase of at most twelve words, with no quotation marks.\n\n"
        "Title: %s\n\nAbstract: %s" % (p["title"], p["abstract"]))
    d = post_json("https://api.anthropic.com/v1/messages",
                  {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  {"model": SUBJECT_MODEL, "max_tokens": 100, "messages": [{"role": "user", "content": prompt}]}, 120)
    s = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    s = re.sub(r"\s+", " ", s).strip().strip("\"'").rstrip(".")
    if not s:
        raise RuntimeError("Claude returned no subject")
    u = d.get("usage", {})
    log("  subject from Claude (%s in / %s out tokens): %s" % (u.get("input_tokens"), u.get("output_tokens"), s))
    return s


# ---------------------------------------------------------------- study and engraving
def make_study(subject, dest):
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_FAI_API_KEY")
    d = post_json("https://openrouter.ai/api/v1/images/generations", {"Authorization": "Bearer " + key},
                  {"model": STUDY_MODEL, "prompt": STUDY_PROMPT + subject + ".", "aspect_ratio": "16:9"})
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(d["data"][0]["b64_json"]))).convert("L")
    if im.size != (1376, 768):
        im = im.resize((1376, 768), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)
    cost = (d.get("usage") or {}).get("cost") or 0
    log("  study %s (cost $%.4f)" % (dest.name, cost))
    return float(cost)


def engrave(study, key, seed, out_art):
    from PIL import Image
    for d in ("band", "tile", "field"):
        (out_art / d).mkdir(parents=True, exist_ok=True)
    jobs = {
        "band": (["--x", "0.68"], BAND_SIZE),
        "tile": (["--w", "720", "--h", "450", "--cell", "9", "--x", "0.5", "--fit", "0.95"], None),
    }
    with tempfile.TemporaryDirectory() as tmp:
        procs = {}
        for kind, (flags, _) in jobs.items():
            still = Path(tmp) / (kind + ".png")
            cmd = [sys.executable, str(ENGRAVE), str(study), str(still), *flags, "--seed", str(seed),
                   "--field", str(out_art / "field" / ("%s-%s.png" % (kind, key)))]
            procs[kind] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), still)
        for kind, (proc, still) in procs.items():
            o = proc.communicate()[0]
            if proc.returncode:
                raise RuntimeError("engrave.py (%s) failed:\n%s" % (kind, o[-800:]))
            im = Image.open(still).convert("RGB")
            if jobs[kind][1] and im.size != jobs[kind][1]:
                im = im.resize(jobs[kind][1], Image.LANCZOS)
            dest = out_art / kind / (key + ".jpg")
            im.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
            log("  %s %s %dx%d %d KB" % (kind, dest.name, im.width, im.height, dest.stat().st_size // 1024))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", nargs="*", default=None, metavar="KEY", help="redo these keys (space or comma separated)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", metavar="DIR", help="write DIR/art/... instead of assets/art/...")
    a = ap.parse_args()

    forced = {k.strip() for v in (a.force or []) for k in v.split(",") if k.strip()}
    out_art = (Path(a.out).resolve() / "art") if a.out else ART
    if a.out and out_art.resolve().is_relative_to(REPO.resolve()) and out_art.resolve() != ART.resolve():
        log("note: --out is inside the repository; keep test output out of commits")

    allp = papers()
    known = {p["key"] for p in allp}
    for k in sorted(forced - known):
        log("warning: no paper with key %s; skipping" % k)
    todo = [p for p in allp if p["key"] in forced or (not forced and not has_art(p["key"]))]
    if not todo:
        log("Every paper already has art. Nothing to do.")
        return 0

    subjects_path = ART / "subjects.json"
    subjects = json.loads(subjects_path.read_text(encoding="utf-8")) if subjects_path.exists() else {}

    if a.dry_run:
        for p in todo:
            s = subjects.get(p["key"])
            log("%s  %s  seed %s  subject: %s" % (p["key"], p["title"], p["seed"],
                                                  s or "(Claude will propose one)"))
        log("Dry run: %d paper(s); nothing written." % len(todo))
        return 0

    missing = []
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_FAI_API_KEY")):
        missing.append("OPENROUTER_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY") and any(p["key"] not in subjects for p in todo):
        missing.append("ANTHROPIC_API_KEY (needed to propose subjects missing from subjects.json)")
    if missing:
        log("NOTICE: not making paper art because these keys are not set: " + "; ".join(missing))
        log("Add them as repository secrets (Settings > Secrets and variables > Actions) and run the workflow again.")
        return 0

    try:
        import numpy, scipy, PIL  # noqa: F401
    except ImportError as e:
        log("error: %s. Install with: pip install numpy scipy pillow" % e)
        return 1

    total, failed = 0.0, []
    for p in todo:
        key = p["key"]
        log("[%s] %s" % (key, p["title"]))
        try:
            subject = subjects.get(key)
            if subject:
                log("  subject from subjects.json: " + subject)
            else:
                subject = propose_subject(p)
                subjects[key] = subject
                dest = (out_art.parent / "art" / "subjects.json") if a.out else subjects_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(dict(sorted(subjects.items())), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            study = out_art / "src" / (key + ".png")
            if key in forced or not study.exists():
                total += make_study(subject, study)
            else:
                log("  reusing study " + study.name)
            engrave(study, key, p["seed"], out_art)
        except Exception as e:  # noqa: BLE001
            log("  FAILED: %s" % e)
            failed.append(key)
    log("Done: %d made, %d failed%s. OpenRouter cost $%.4f." % (len(todo) - len(failed), len(failed),
                                                                 (" (" + ", ".join(failed) + ")") if failed else "", total))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
