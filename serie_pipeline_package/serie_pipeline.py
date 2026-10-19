#!/usr/bin/env python3
"""Pipeline d'episode: script annote -> alignement force -> Manim -> MP4 + voix originale.

Requires: ffmpeg/ffprobe, whisperx, and Manim Community in the same environment.
The animation template should expose a C = dict(...) and END_T = ... like the provided scene.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
TIMESTAMP_LINE_RE = re.compile(r"\*\*\[\s*\d+(?:\.\d+)?\s*\]\*\*")


def run_checked(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, env=env)


def require_program(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Programme introuvable: {name}. Installe-le avec: sudo apt install ffmpeg")
    return exe


def audio_duration_seconds(audio_path: Path) -> float:
    ffprobe = require_program("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        check=True, capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Impossible de lire la duree de {audio_path}") from exc


def clean_annotated_script(path: Path) -> str:
    """Retire timestamps, markup gras et annotations entre parentheses du script fourni."""
    raw = path.read_text(encoding="utf-8-sig")
    first_marker = TIMESTAMP_LINE_RE.search(raw)
    if first_marker:
        raw = raw[first_marker.start():]
    # Les parenthèses marquées en italique sont des cues/notes, pas du texte parlé.
    raw = re.sub(r"\*\([^)]*\)\*", " ", raw)
    # Retirer les horodatages de début de ligne comme **[6.0]**.
    raw = TIMESTAMP_LINE_RE.sub(" ", raw)
    # Enlever le gras/italique Markdown sans effacer le texte mis en valeur.
    raw = raw.replace("**", "")
    raw = raw.replace("__", "")
    raw = re.sub(r"(?<!\*)\*(?!\*)", "", raw)
    raw = re.sub(r"(?m)^\s*---\s*$", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    # Fichier vide ou compose seulement de notes => erreur explicite.
    if len(WORD_RE.findall(raw)) < 5:
        raise RuntimeError("Le texte parle est trop court ou le format du script annote n'a pas ete reconnu.")
    return raw


def extract_bold_spans(path: Path) -> list[str]:
    """Retourne les segments en gras dans l'ordre du script; les timestamps ne comptent pas."""
    raw = path.read_text(encoding="utf-8-sig")
    first_marker = TIMESTAMP_LINE_RE.search(raw)
    if first_marker:
        raw = raw[first_marker.start():]
    raw = re.sub(r"\*\([^)]*\)\*", " ", raw)
    raw = TIMESTAMP_LINE_RE.sub(" ", raw)
    spans = [piece.strip() for piece in re.findall(r"\*\*(.*?)\*\*", raw, flags=re.S)]
    return [piece for piece in spans if normalise_tokens(piece)]


def normalise_tokens(text: str) -> list[str]:
    return [m.group(0).lower().replace("’", "'") for m in WORD_RE.finditer(text)]


def analyse_et_align_audio(
    audio_path: Path, transcript: str, language: str, device: str, threads: int,
    asr_model_name: str | None = "base.en", check_transcription: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    """Transcrit facultativement avec Whisper, puis force-aligne le texte fourni."""
    try:
        import gc
        import torch
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX n'est pas installe dans cet environnement. Lance: pip install whisperx"
        ) from exc

    if device == "cpu":
        try:
            torch.set_num_threads(max(1, threads))
        except Exception:
            pass
    print(f"Chargement du MP3 et alignement force WhisperX ({language}, {device})...", flush=True)
    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000.0
    asr_transcript: str | None = None

    if check_transcription and asr_model_name:
        print(f"Etape 1/2: transcription de controle avec Whisper ({asr_model_name})...", flush=True)
        compute_type = "int8" if device == "cpu" else "float16"
        try:
            asr_model = whisperx.load_model(
                asr_model_name, device, compute_type=compute_type, language=language
            )
        except TypeError:
            asr_model = whisperx.load_model(asr_model_name, device, compute_type=compute_type)
        try:
            recognized = asr_model.transcribe(audio, batch_size=4, language=language)
        except TypeError:
            recognized = asr_model.transcribe(audio, batch_size=4)
        asr_transcript = " ".join(str(seg.get("text", "")).strip() for seg in recognized.get("segments", [])).strip()
        del asr_model
        gc.collect()
        print("Transcription de controle terminee.", flush=True)

    print("Etape 2/2: alignement du texte de reference mot par mot...", flush=True)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    reference = [{"start": 0.0, "end": audio_duration, "text": transcript}]
    try:
        aligned = whisperx.align(reference, align_model, metadata, audio, device)
    except TypeError:
        # Compatibilite avec certaines versions de WhisperX.
        aligned = whisperx.align(reference, align_model, metadata, audio, device=device)

    words = aligned.get("word_segments") or []
    if not words:
        for segment in aligned.get("segments", []):
            words.extend(segment.get("words", []))
    cleaned: list[dict[str, Any]] = []
    for item in words:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        start, end = item.get("start"), item.get("end")
        cleaned.append({
            "word": word,
            "start": float(start) if start is not None else None,
            "end": float(end) if end is not None else None,
            "score": float(item["score"]) if item.get("score") is not None else None,
        })
    if len(cleaned) < 5:
        raise RuntimeError("WhisperX n'a pas renvoye assez de mots alignes. Verifie la langue et l'enregistrement.")
    return cleaned, asr_transcript


def phrase_hits(word_segments: list[dict[str, Any]], phrase: str) -> list[int]:
    target = normalise_tokens(phrase)
    actual = [normalise_tokens(str(w.get("word", ""))) for w in word_segments]
    # Un element whisperx correspond normalement a un mot. On ignore ici les segments
    # anormaux comportant plusieurs tokens en les aplatissant avec leurs indices d'origine.
    flat: list[str] = []
    owner: list[int] = []
    for i, toks in enumerate(actual):
        for tok in toks:
            flat.append(tok)
            owner.append(i)
    hits: list[int] = []
    if not target:
        return hits
    for i in range(0, len(flat) - len(target) + 1):
        if flat[i:i + len(target)] == target:
            # The phrase can only anchor to simple one-token-per-word spans reliably.
            hits.append(owner[i])
    # If the same target begins inside one multi-token segment, remove duplicate owners.
    return list(dict.fromkeys(hits))


def align_bold_spans(word_segments: list[dict[str, Any]], bold_spans: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Relie chaque segment en gras au meme segment audio, dans l'ordre du texte."""
    aligned_spans: list[dict[str, Any]] = []
    errors: list[str] = []
    cursor = 0
    for index, span in enumerate(bold_spans):
        target = normalise_tokens(span)
        actual = [normalise_tokens(str(w.get("word", ""))) for w in word_segments]
        flat: list[str] = []
        owner: list[int] = []
        for i, toks in enumerate(actual):
            for tok in toks:
                flat.append(tok)
                owner.append(i)
        found = None
        for i in range(cursor, len(flat) - len(target) + 1):
            if flat[i:i + len(target)] == target:
                found = (i, owner[i])
                break
        if found is None:
            errors.append(f"Segment en gras #{index} introuvable dans l'alignement: {span!r}")
            aligned_spans.append({"text": span, "start_idx": None, "word_count": len(target)})
            continue
        flat_start, word_start = found
        # Les sorties normales de WhisperX contiennent une entree par mot. Si ce n'est
        # pas le cas, on signale plus loin un mapping douteux au lieu de deviner silencieusement.
        if len(set(owner[flat_start:flat_start + len(target)])) != len(target):
            errors.append(f"Segment en gras #{index} correspond a un segment audio composite: {span!r}")
        aligned_spans.append({"text": span, "start_idx": word_start, "word_count": len(target)})
        cursor = flat_start + len(target)
    return aligned_spans, errors


def create_cues(word_segments: list[dict[str, Any]], cue_map: dict[str, Any], template_source: str, bold_spans: list[str]) -> tuple[dict[str, float], list[str]]:
    expected_keys = extract_c_keys(template_source)
    rules = cue_map.get("cues", cue_map)
    aligned_bold, errors = align_bold_spans(word_segments, bold_spans)
    output: dict[str, float] = {}

    absent = [key for key in expected_keys if key not in rules]
    extra = [key for key in rules if key not in expected_keys]
    if absent:
        errors.append("Cues sans regle de mapping: " + ", ".join(absent))
    if extra:
        errors.append("Regles qui ne correspondent pas a une cle C: " + ", ".join(extra))

    for key in expected_keys:
        if key not in rules:
            continue
        rule = rules[key]
        offset = float(rule.get("offset", 0.0))
        edge = str(rule.get("edge", "start")).lower()
        if edge not in {"start", "end"}:
            errors.append(f"{key}: edge doit etre 'start' ou 'end'")
            continue

        if "bold_index" in rule:
            bold_index = int(rule["bold_index"])
            if bold_index < 0 or bold_index >= len(aligned_bold):
                errors.append(f"{key}: bold_index={bold_index} hors limites (gras disponibles: {len(aligned_bold)})")
                continue
            anchor = aligned_bold[bold_index]
            word_index = int(rule.get("word_index", 0))
            if anchor.get("start_idx") is None or word_index < 0 or word_index >= anchor["word_count"]:
                errors.append(f"{key}: ancre en gras #{bold_index} ou word_index={word_index} invalide")
                continue
            target_idx = int(anchor["start_idx"]) + word_index
        else:
            phrase = str(rule.get("phrase", "")).strip()
            hits = phrase_hits(word_segments, phrase)
            occurrence = int(rule.get("occurrence", 1))
            if not hits or occurrence < 1 or occurrence > len(hits):
                errors.append(f"{key}: phrase introuvable ou occurrence invalide: {phrase!r} (trouve {len(hits)})")
                continue
            phrase_words = normalise_tokens(phrase)
            word_index = int(rule.get("word_index", 0))
            if not phrase_words or word_index < 0 or word_index >= len(phrase_words):
                errors.append(f"{key}: word_index={word_index} invalide pour {phrase!r}")
                continue
            target_idx = hits[occurrence - 1] + word_index

        if target_idx >= len(word_segments):
            errors.append(f"{key}: index de mot hors limites")
            continue
        target_word = word_segments[target_idx]
        raw_time = target_word.get(edge)
        if raw_time is None:
            errors.append(f"{key}: horodatage manquant sur {target_word.get('word')!r}")
            continue
        output[key] = round(max(0.0, float(raw_time) + offset), 3)

    return output, errors


def extract_c_keys(source: str) -> list[str]:
    match = re.search(r"\bC\s*=\s*dict\((.*?)\n\)", source, flags=re.S)
    if not match:
        raise RuntimeError("Impossible de trouver `C = dict(...)` dans le fichier Manim.")
    keys = re.findall(r"([A-Za-z_]\w*)\s*=", match.group(1))
    if not keys:
        raise RuntimeError("Le dictionnaire C est vide ou son format n'est pas reconnu.")
    return keys


def patch_animation_source(template_path: Path, destination: Path) -> str:
    source = template_path.read_text(encoding="utf-8")
    if "MANIM_CUES_JSON" in source:
        destination.write_text(source, encoding="utf-8")
        return source
    loader = '''\n\n# Horaires generes automatiquement par serie_pipeline.py\n_cues_json_path = os.environ.get("MANIM_CUES_JSON")\nif _cues_json_path:\n    import json as _json\n    with open(_cues_json_path, "r", encoding="utf-8") as _cue_file:\n        _generated_cues = _json.load(_cue_file)\n    C.update({k: float(v) for k, v in _generated_cues["C"].items()})\n    END_T = float(_generated_cues["END_T"])\n'''
    pattern = re.compile(r"(?m)^END_T\s*=\s*[^\n]+")
    patched, count = pattern.subn(lambda m: m.group(0) + loader, source, count=1)
    if count != 1:
        raise RuntimeError("Impossible d'inserer le chargeur de cues apres END_T dans le fichier d'animation.")
    destination.write_text(patched, encoding="utf-8")
    return patched


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER simple sur les mots normalises; sert d'avertissement, pas de verdict absolu."""
    ref = normalise_tokens(reference)
    hyp = normalise_tokens(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (r != h),
            ))
        previous = current
    return previous[-1] / len(ref)


def write_reports(output_dir: Path, transcript: str, words: list[dict[str, Any]], cues: dict[str, float], end_t: float, errors: list[str], audio_duration: float, asr_transcript: str | None = None) -> None:
    (output_dir / "script_clean.txt").write_text(transcript + "\n", encoding="utf-8")
    with (output_dir / "word_times.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start", "end", "score"])
        writer.writeheader()
        writer.writerows(words)
    report = [
        f"Duree audio: {audio_duration:.3f} s",
        f"Nombre de mots retournes par l'alignement: {len(words)}",
        f"Fin prevue de la scene Manim: {end_t:.3f} s",
    ]
    if asr_transcript is not None:
        wer = word_error_rate(transcript, asr_transcript)
        report += [
            f"WER indicatif Whisper vs script: {wer:.1%}",
            "Transcription de controle Whisper: " + asr_transcript,
        ]
        (output_dir / "whisper_transcription.txt").write_text(asr_transcript + "\n", encoding="utf-8")
        if wer > 0.12:
            report.append("ATTENTION: Whisper differe sensiblement du script. Verifie la diction avant le rendu final.")
    report += ["", "TIMINGS C:"]
    report.extend(f"  {key:16s} {value:8.3f} s" for key, value in cues.items())
    report += ["", "VALIDATION:"]
    if errors:
        report.extend("  ERREUR: " + err for err in errors)
    else:
        report.append("  Toutes les ancres ont ete trouvees.")
    if sum(1 for w in words if w.get("start") is None or w.get("end") is None):
        report.append("  ATTENTION: certains mots n'ont pas d'horodatage.")
    (output_dir / "alignment_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def render_and_mux(animation_script: Path, audio_path: Path, cue_json_path: Path, output_dir: Path, scene: str) -> Path:
    ffmpeg = require_program("ffmpeg")
    media_dir = output_dir / "manim_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    silent_name = "animation_silent.mp4"
    manim_cli = shutil.which("manim")
    if manim_cli:
        cmd = [manim_cli]
    else:
        cmd = [sys.executable, "-m", "manim"]
    cmd += [
        "render", "--disable_caching", "--media_dir", str(media_dir),
        "--format", "mp4", "--output_file", silent_name,
        str(animation_script), scene,
    ]
    env = os.environ.copy()
    env["MANIM_CUES_JSON"] = str(cue_json_path.resolve())
    run_checked(cmd, env=env)

    candidates = [p for p in media_dir.rglob(silent_name) if "partial_movie_files" not in p.parts]
    if not candidates:
        candidates = [p for p in media_dir.rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if not candidates:
        raise RuntimeError(f"Manim a termine sans produire de MP4. Regarde les logs dans {media_dir}")
    silent_video = max(candidates, key=lambda p: p.stat().st_mtime)
    copied_silent = output_dir / silent_name
    if silent_video.resolve() != copied_silent.resolve():
        shutil.copy2(silent_video, copied_silent)

    final_path = output_dir / "episode_final.mp4"
    mux_cmd = [
        ffmpeg, "-y", "-i", str(copied_silent), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest",
        "-movflags", "+faststart", str(final_path),
    ]
    run_checked(mux_cmd)
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatise l'alignement de la voix, les cues Manim et l'assemblage video/audio."
    )
    parser.add_argument("script", type=Path, help="Script annote: mots declencheurs en gras, temps entre parentheses facultatifs")
    parser.add_argument("audio", type=Path, help="Voix off (MP3, WAV, M4A, etc.)")
    parser.add_argument("animation", type=Path, help="Fichier source de la scene Manim (Python; .txt accepte aussi)")
    parser.add_argument("--cue-map", type=Path, default=Path("cue_map.json"), help="JSON reliant les cles C aux phrases du script")
    parser.add_argument("--scene", default="Unguided", help="Nom de la classe Scene dans le fichier Manim")
    parser.add_argument("--output-dir", type=Path, default=Path("build_episode"), help="Dossier de sortie")
    parser.add_argument("--language", default="en", help="Code de langue WhisperX pour le modele d'alignement")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--asr-model", default="base.en", help="Modele Whisper de controle (ex. base.en ou small.en)")
    parser.add_argument("--skip-asr-check", action="store_true", help="Saute la transcription de controle et fait seulement l'alignement force")
    parser.add_argument("--tail", type=float, default=1.0, help="Silence final minimal a laisser apres la duree audio (s)")
    parser.add_argument("--align-only", action="store_true", help="Genere les horodatages et le dictionnaire C sans lancer Manim ni FFmpeg")
    args = parser.parse_args()

    script_path = args.script.expanduser().resolve()
    audio_path = args.audio.expanduser().resolve()
    animation_path = args.animation.expanduser().resolve()
    cue_map_path = args.cue_map.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (script_path, audio_path, animation_path, cue_map_path):
        if not path.is_file():
            parser.error(f"Fichier introuvable: {path}")
    if args.threads < 1:
        parser.error("--threads doit etre >= 1")
    if args.tail < 0:
        parser.error("--tail ne peut pas etre negatif")
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = audio_duration_seconds(audio_path)
    template_source = animation_path.read_text(encoding="utf-8")
    try:
        expected_keys = extract_c_keys(template_source)
    except RuntimeError as exc:
        parser.error(str(exc))
    transcript = clean_annotated_script(script_path)
    bold_spans = extract_bold_spans(script_path)
    cue_map = json.loads(cue_map_path.read_text(encoding="utf-8"))

    print(f"Audio: {duration:.2f} s | texte: {len(normalise_tokens(transcript))} mots")
    try:
        words, asr_transcript = analyse_et_align_audio(
            audio_path, transcript, args.language, args.device, args.threads,
            asr_model_name=args.asr_model, check_transcription=not args.skip_asr_check,
        )
    except Exception as exc:
        print(f"Erreur pendant l'analyse/alignement audio: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    cues, errors = create_cues(words, cue_map, template_source, bold_spans)
    # `self.at()` ne peut pas revenir en arriere dans le temps. Une inversion ne bloque pas
    # le rendu, mais sera signalee car les appels Manim pourront alors partir en retard.
    cue_order = extract_c_keys(template_source)
    order_warnings = []
    for previous_key, current_key in zip(cue_order, cue_order[1:]):
        if previous_key in cues and current_key in cues and cues[current_key] < cues[previous_key] - 0.05:
            order_warnings.append(
                f"Ordre des cues inverse: {previous_key}={cues[previous_key]:.3f}s puis "
                f"{current_key}={cues[current_key]:.3f}s. Verifie les ancres/offsets."
            )

    if errors:
        write_reports(output_dir, transcript, words, cues, 0.0, errors, duration, asr_transcript)
        print("\nARRET: l'alignement ou le mapping contient des erreurs. Aucun rendu lance.")
        print("Consulte:", output_dir / "alignment_report.txt")
        for err in errors:
            print(" -", err)
        return 2

    if "loop" in cues:
        computed_end = max(duration + args.tail, cues["loop"] + 1.0)
    else:
        latest = max((w.get("end") or 0.0 for w in words), default=0.0)
        computed_end = max(duration + args.tail, latest + args.tail)
    computed_end = round(computed_end, 3)
    cues_json = {"C": cues, "END_T": computed_end, "audio_duration": round(duration, 3)}
    cue_json_path = output_dir / "cues_generated.json"
    cue_json_path.write_text(json.dumps(cues_json, indent=2), encoding="utf-8")
    write_reports(output_dir, transcript, words, cues, computed_end, errors, duration, asr_transcript)
    if order_warnings:
        report_path = output_dir / "alignment_report.txt"
        with report_path.open("a", encoding="utf-8") as report_file:
            report_file.write("\nAVERTISSEMENTS D'ORDRE (les appels Manim peuvent etre en retard):\n")
            report_file.write("\n".join("  " + warning for warning in order_warnings) + "\n")
        print("\nAttention: ordre de certains cues a verifier:")
        for warning in order_warnings:
            print(" -", warning)
    render_script = output_dir / "animation_with_cues.py"
    patch_animation_source(animation_path, render_script)

    # Signale les evenements proches: les animations Manim ont elles-memes une duree.
    ordered = sorted(cues.items(), key=lambda item: item[1])
    close = [(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)) if ordered[i][1] - ordered[i - 1][1] < 0.22]
    if close:
        print("\nAttention: certains cues sont espaces de moins de 0,22 s:")
        for previous, current in close[:15]:
            print(f"  {previous[0]} -> {current[0]}: {current[1] - previous[1]:.3f} s")
        print("Ces evenements peuvent etre retardes par la duree des self.play() dans Manim.")

    print("\nDictionnaire C genere:", cue_json_path)
    print("Rapport:", output_dir / "alignment_report.txt")
    if args.align_only:
        print("Mode --align-only: rendu Manim non lance.")
        return 0

    try:
        final_path = render_and_mux(render_script, audio_path, cue_json_path, output_dir, args.scene)
    except subprocess.CalledProcessError as exc:
        print(f"\nLa commande a echoue (code {exc.returncode}). Le JSON et le rapport sont conserves.", file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(f"\nErreur de rendu: {exc}", file=sys.stderr)
        return 1
    print("\nTermine:", final_path)
    print("La voix off d'origine a ete conservee. Un silence est ajoute seulement si l'animation dure plus longtemps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
