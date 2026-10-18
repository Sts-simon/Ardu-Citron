#!/usr/bin/env python3
"""Reconstruit une voix off en replaçant ses phrases selon script_timeline.

Le programme transcrit l'audio avec Whisper, identifie les phrases, puis recolle
les extraits dans l'ordre du script aux horaires demandes. Si des horaires se
chevauchent, la phrase suivante est decalee pour eviter de couper ou superposer
la voix. Les sous-titres sont ensuite generes pour le nouvel audio.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[''][A-Za-z0-9]+)*")


def normalize_word(text: str) -> str:
    """Normalise un mot pour comparer script et transcription."""
    return re.sub(r"[^a-z0-9]", "", text.casefold().replace("'", "'"))


def tokenize(text: str) -> list[str]:
    return [normalize_word(match.group(0)) for match in WORD_RE.finditer(text) if normalize_word(match.group(0))]


def load_timeline(path: Path) -> list[dict[str, Any]]:
    """Lit script_timeline depuis un fichier texte contenant une affectation Python."""
    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(
            "Le fichier TXT doit contenir une liste Python valide, par exemple "
            "`script_timeline = [{\"start\": 0, \"end\": 1, \"text\": \"Hello\"}]`."
        ) from exc

    timeline = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "script_timeline"
            for target in node.targets
        ):
            timeline = ast.literal_eval(node.value)
            break
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "script_timeline":
            timeline = ast.literal_eval(node.value)
            break

    if timeline is None:
        # Permet aussi un fichier qui contient uniquement la liste, sans variable.
        try:
            timeline = ast.literal_eval(source.strip())
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Impossible de trouver `script_timeline = [...]` dans le fichier TXT.") from exc

    if not isinstance(timeline, list) or not timeline:
        raise ValueError("script_timeline doit etre une liste non vide.")

    cleaned = []
    for index, item in enumerate(timeline):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"L'entree {index} doit contenir au minimum la cle text.")
        if not tokenize(item["text"]):
            raise ValueError(f"L'entree {index} ne contient aucun mot : {item['text']!r}")
        cleaned.append({
            "start": float(item.get("start", 0.0)),
            "end": float(item.get("end", item.get("start", 0.0))),
            "text": item["text"],
        })
    return cleaned


def flatten_script(timeline: list[dict[str, Any]]) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Aplatit les mots du script et memorise la ligne de sous-titres de chaque mot."""
    words: list[str] = []
    owners: list[int] = []
    ranges: list[tuple[int, int]] = []
    for line_index, item in enumerate(timeline):
        start_index = len(words)
        line_words = tokenize(item["text"])
        words.extend(line_words)
        owners.extend([line_index] * len(line_words))
        ranges.append((start_index, len(words)))
    return words, owners, ranges


def align_words(reference: list[str], hypothesis: list[str]) -> dict[int, list[int]]:
    """Aligne les mots du script sur les mots Whisper par programmation dynamique.

    Les mots de script absents de la transcription et les mots ajoutes par Whisper
    sont ignores. Gere aussi les acronymes prononces lettre par lettre, comme CCIP.
    """
    n, m = len(reference), len(hypothesis)
    gap_ref = -1.0
    gap_hyp = -1.0
    mismatch = -2.4

    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_ref
        back[i][0] = (i - 1, 0, "D", 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_hyp
        back[0][j] = (0, j - 1, "I", 0)

    for i in range(1, n + 1):
        ref = reference[i - 1]
        for j in range(1, m + 1):
            hyp = hypothesis[j - 1]
            ratio = SequenceMatcher(None, ref, hyp).ratio()
            if ref == hyp:
                match_score = 4.0
            elif min(len(ref), len(hyp)) >= 3 and ratio >= 0.70:
                match_score = 1.0 + ratio
            else:
                match_score = mismatch

            candidates: list[tuple[float, tuple[int, int, str, int]]] = [
                (dp[i - 1][j - 1] + match_score, (i - 1, j - 1, "M", 1)),
                (dp[i - 1][j] + gap_ref, (i - 1, j, "D", 0)),
                (dp[i][j - 1] + gap_hyp, (i, j - 1, "I", 0)),
            ]

            # Cas frequent avec CCIP/CCRP, parfois transcrits  C C I P .
            if len(ref) >= 2:
                for count in range(2, min(6, j) + 1):
                    group = hypothesis[j - count:j]
                    if all(len(word) == 1 for word in group) and "".join(group) == ref:
                        candidates.append((
                            dp[i - 1][j - count] + 4.5,
                            (i - 1, j - count, "G", count),
                        ))

            best_score, best_pointer = max(candidates, key=lambda candidate: candidate[0])
            dp[i][j] = best_score
            back[i][j] = best_pointer

    mapping: dict[int, list[int]] = {}
    i, j = n, m
    while i > 0 or j > 0:
        pointer = back[i][j]
        if pointer is None:
            break
        previous_i, previous_j, operation, count = pointer
        if operation == "M":
            ratio = SequenceMatcher(None, reference[i - 1], hypothesis[j - 1]).ratio()
            if reference[i - 1] == hypothesis[j - 1] or ratio >= 0.70:
                mapping[i - 1] = [j - 1]
        elif operation == "G":
            mapping[i - 1] = list(range(j - count, j))
        i, j = previous_i, previous_j

    return mapping


def build_word_times(
    reference: list[str],
    hypothesis_words: list[dict[str, Any]],
    mapping: dict[int, list[int]],
    duration: float,
) -> tuple[list[float], list[float], float]:
    """Attribue un intervalle a chaque mot du script, meme si Whisper en a rate un."""
    count = len(reference)
    starts: list[float | None] = [None] * count
    ends: list[float | None] = [None] * count
    known_durations: list[float] = []

    for ref_index, hyp_indices in mapping.items():
        if not hyp_indices:
            continue
        first = hypothesis_words[hyp_indices[0]]
        last = hypothesis_words[hyp_indices[-1]]
        start = max(0.0, float(first["start"]))
        end = max(start, float(last["end"]))
        starts[ref_index] = start
        ends[ref_index] = end
        if end > start:
            known_durations.append(end - start)

    average = statistics.median(known_durations) if known_durations else 0.25
    average = min(0.45, max(0.12, average))

    # Le programme principal verifie que l'alignement contient des correspondances.
    known_indices = [i for i, value in enumerate(starts) if value is not None]
    if not known_indices:
        raise ValueError("Aucun mot du script ne correspond a la transcription Whisper.")

    # Interpole les sequences de mots non reconnus entre deux mots reconnus.
    cursor = 0
    while cursor < count:
        if starts[cursor] is not None:
            cursor += 1
            continue
        run_start = cursor
        while cursor < count and starts[cursor] is None:
            cursor += 1
        run_end = cursor  # exclusif
        missing_count = run_end - run_start
        left = run_start - 1
        right = run_end

        if left >= 0 and right < count:
            left_edge = float(ends[left])
            right_edge = float(starts[right])
            if right_edge > left_edge + 0.01:
                width = (right_edge - left_edge) / (missing_count + 1)
                for offset in range(missing_count):
                    idx = run_start + offset
                    starts[idx] = left_edge + width * offset
                    ends[idx] = left_edge + width * (offset + 1)
            else:
                # Peu d'espace entre les ancrages : on place les mots autour de leur milieu.
                center = (left_edge + right_edge) / 2.0
                first_start = center - average * missing_count / 2.0
                for offset in range(missing_count):
                    idx = run_start + offset
                    starts[idx] = max(0.0, first_start + offset * average)
                    ends[idx] = min(duration, float(starts[idx]) + average)
        elif right < count:
            right_edge = float(starts[right])
            local_width = min(average, right_edge / max(1, missing_count)) if right_edge > 0 else average
            first_start = max(0.0, right_edge - local_width * missing_count)
            for offset in range(missing_count):
                idx = run_start + offset
                starts[idx] = first_start + offset * local_width
                ends[idx] = min(right_edge, first_start + (offset + 1) * local_width)
        else:
            left_edge = float(ends[left]) if left >= 0 else 0.0
            for offset in range(missing_count):
                idx = run_start + offset
                starts[idx] = min(duration, left_edge + offset * average)
                ends[idx] = min(duration, left_edge + (offset + 1) * average)

    # Type narrowing: toutes les positions ont ete remplies a ce stade.
    final_starts = [max(0.0, min(duration, float(value))) for value in starts]
    final_ends = [max(final_starts[i], min(duration, float(ends[i]))) for i in range(count)]
    return final_starts, final_ends, average


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path, timeline: list[dict[str, Any]]) -> None:
    chunks = []
    number = 0
    for item in timeline:
        if item.get("_audio_missing"):
            continue
        number += 1
        chunks.append(
            f"{number}\n{format_srt_time(item['start'])} --> {format_srt_time(item['end'])}\n{item['text']}"
        )
    path.write_text(("\n\n".join(chunks) + "\n") if chunks else "", encoding="utf-8")


def write_python_timeline(path: Path, timeline: list[dict[str, Any]]) -> None:
    lines = ["script_timeline = ["]
    for item in timeline:
        if item.get("_audio_missing"):
            continue
        lines.append(
            "    {"
            f"'start': {item['start']:.3f}, "
            f"'end': {item['end']:.3f}, "
            f"'text': {item['text']!r}"
            "},"
        )
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_audio(
    audio_path: Path,
    timeline: list[dict[str, Any]],
    line_ranges: list[tuple[int, int]],
    hypothesis_words: list[dict[str, Any]],
    mapping: dict[int, list[int]],
    output_dir: Path,
    minimum_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path, float]:
    """Decoupe la voix originale en phrases et les recolle aux horaires du script."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "Dependance manquante pour reconstruire l'audio. Lance : pip install pydub"
        ) from exc

    source = AudioSegment.from_file(str(audio_path))
    source_duration = len(source) / 1000.0
    if len(source) == 0:
        raise RuntimeError("Le fichier audio est vide ou illisible.")

    # Horodatages audio reels des mots associes a chaque ligne du script.
    source_spans: list[tuple[float, float] | None] = []
    for first_word, after_last_word in line_ranges:
        matched_indices: list[int] = []
        for reference_index in range(first_word, after_last_word):
            matched_indices.extend(mapping.get(reference_index, []))
        matched_indices = sorted(set(matched_indices))
        if not matched_indices:
            source_spans.append(None)
            continue

        start = min(float(hypothesis_words[j]["start"]) for j in matched_indices)
        end = max(float(hypothesis_words[j]["end"]) for j in matched_indices)
        start = max(0.0, min(source_duration, start))
        end = max(start, min(source_duration, end))
        source_spans.append((start, end) if end - start >= 0.025 else None)

    output_items: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    clips: list[tuple[AudioSegment, int]] = []
    previous_end = 0.0
    has_previous_clip = False

    for line_index, item in enumerate(timeline):
        span = source_spans[line_index]
        requested_start = max(0.0, float(item["start"]))
        if span is None:
            output_items.append({
                "start": round(requested_start, 3),
                "end": round(float(item["end"]), 3),
                "text": item["text"],
                "_audio_missing": True,
            })
            schedule.append({
                "line": line_index + 1,
                "text": item["text"],
                "matched_in_source": False,
                "requested_start": round(requested_start, 3),
                "output_start": None,
                "output_end": None,
                "shift_seconds": None,
            })
            continue

        source_start, source_end = span
        clip = source[int(round(source_start * 1000)):int(round(source_end * 1000))]
        if len(clip) < 25:
            output_items.append({
                "start": round(requested_start, 3),
                "end": round(float(item["end"]), 3),
                "text": item["text"],
                "_audio_missing": True,
            })
            schedule.append({
                "line": line_index + 1,
                "text": item["text"],
                "matched_in_source": False,
                "requested_start": round(requested_start, 3),
                "output_start": None,
                "output_end": None,
                "shift_seconds": None,
            })
            continue

        # Un fondu tres court evite les clics numeriques aux points de decoupe.
        fade_ms = min(4, max(1, len(clip) // 8))
        if len(clip) > fade_ms * 2:
            clip = clip.fade_in(fade_ms).fade_out(fade_ms)

        output_start = requested_start
        if has_previous_clip:
            output_start = max(output_start, previous_end + max(0.0, minimum_gap))
        output_end = output_start + len(clip) / 1000.0
        start_ms = int(round(output_start * 1000))
        clips.append((clip, start_ms))
        output_items.append({
            "start": round(output_start, 3),
            "end": round(output_end, 3),
            "text": item["text"],
        })
        schedule.append({
            "line": line_index + 1,
            "text": item["text"],
            "matched_in_source": True,
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "requested_start": round(requested_start, 3),
            "output_start": round(output_start, 3),
            "output_end": round(output_end, 3),
            "shift_seconds": round(output_start - requested_start, 3),
        })
        previous_end = output_end
        has_previous_clip = True

    if not clips:
        raise RuntimeError(
            "Aucun extrait audio n'a pu etre associe au script. Verifie la transcription et le TXT."
        )

    total_duration = max(
        max(start_ms / 1000.0 + len(clip) / 1000.0 for clip, start_ms in clips),
        max(float(item["end"]) for item in timeline),
    )
    mixed = AudioSegment.silent(
        duration=int(round(total_duration * 1000)),
        frame_rate=source.frame_rate,
    ).set_channels(source.channels).set_sample_width(source.sample_width)
    for clip, start_ms in clips:
        mixed = mixed.overlay(clip, position=start_ms)

    output_audio = output_dir / "voiceover_aligned.mp3"
    mixed.export(str(output_audio), format="mp3", bitrate="192k")
    return output_items, schedule, output_audio, len(mixed) / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcrit une voix off avec Whisper puis synchronise les lignes du script."
    )
    parser.add_argument("script_txt", type=Path, help="TXT contenant script_timeline = [...]")
    parser.add_argument("audio", type=Path, help="Fichier audio de la voix off")
    parser.add_argument("--model", default="base.en", help="Modele Whisper (defaut : base.en ; essayer small.en pour plus de precision)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Calcul CPU par defaut, plus compatible avec les anciens GPU")
    parser.add_argument("--language", default="en", help="Langue parlee (defaut : en)")
    parser.add_argument("--threads", type=int, default=4, help="Nombre de threads CPU (defaut : 4)")
    parser.add_argument("--output-dir", type=Path, default=Path("aligned_output"), help="Dossier des fichiers generes")
    parser.add_argument("--min-gap", type=float, default=0.03, help="Silence minimal entre deux extraits audio (defaut : 0.03 s)")
    args = parser.parse_args()

    if not args.script_txt.is_file():
        parser.error(f"Fichier TXT introuvable : {args.script_txt}")
    if not args.audio.is_file():
        parser.error(f"Fichier audio introuvable : {args.audio}")

    try:
        timeline = load_timeline(args.script_txt)
    except Exception as exc:
        print(f"Erreur de lecture du script : {exc}", file=sys.stderr)
        return 2

    try:
        import whisper_timestamped as whisper
        import torch
    except ImportError as exc:
        print(
            "Dependance manquante. Active ton environnement virtuel puis execute :\n"
            "  pip install whisper-timestamped pydub\n"
            "Verifie aussi que ffmpeg est installe.\n"
            f"Detail : {exc}",
            file=sys.stderr,
        )
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA n'est pas disponible. Passage en mode CPU.")
        args.device = "cpu"
    try:
        torch.set_num_threads(max(1, args.threads))
    except Exception:
        pass

    print(f"Lecture du script : {len(timeline)} lignes de sous-titres")
    print(f"Chargement de Whisper ({args.model}) sur {args.device}...")
    try:
        model = whisper.load_model(args.model, device=args.device)
        audio = whisper.load_audio(str(args.audio))
        audio_duration = len(audio) / 16000.0
        print(f"Analyse de la voix off ({audio_duration:.1f} s)...")
        result = whisper.transcribe(model, audio, language=args.language)
    except Exception as exc:
        print(f"Erreur pendant l'analyse audio : {exc}", file=sys.stderr)
        return 1

    hypothesis_words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            text = str(word.get("text", "")).strip()
            norm = normalize_word(text)
            if not norm:
                continue
            hypothesis_words.append({
                "text": text,
                "norm": norm,
                "start": float(word.get("start", segment.get("start", 0.0))),
                "end": float(word.get("end", segment.get("end", 0.0))),
                "probability": word.get("confidence", word.get("probability")),
            })

    if not hypothesis_words:
        print("Whisper n'a renvoye aucun horodatage par mot. Verifie le fichier audio.", file=sys.stderr)
        return 1

    reference_words, owners, line_ranges = flatten_script(timeline)
    hypothesis_norms = [word["norm"] for word in hypothesis_words]
    mapping = align_words(reference_words, hypothesis_norms)
    if not mapping:
        print("Aucun mot du script ne correspond a la transcription. Verifie la langue, le fichier audio et le script.", file=sys.stderr)
        print(f"Transcription obtenue : {result.get('text', '')}", file=sys.stderr)
        return 1
    starts, ends, estimated_word_duration = build_word_times(
        reference_words, hypothesis_words, mapping, audio_duration
    )

    source_timeline: list[dict[str, Any]] = []
    line_match_stats: list[tuple[int, int]] = []
    for line_index, (first_word, after_last_word) in enumerate(line_ranges):
        line_starts = [starts[i] for i in range(first_word, after_last_word)]
        line_ends = [ends[i] for i in range(first_word, after_last_word)]
        matched = sum(1 for i in range(first_word, after_last_word) if i in mapping)
        total = after_last_word - first_word
        line_start = max(0.0, min(line_starts))
        line_end = min(audio_duration, max(line_ends))
        if line_end <= line_start:
            line_end = min(audio_duration, line_start + max(0.15, estimated_word_duration))
        source_timeline.append({"start": round(line_start, 3), "end": round(line_end, 3), "text": timeline[line_index]["text"]})
        line_match_stats.append((matched, total))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = args.output_dir / "subtitles_aligned.srt"
    py_path = args.output_dir / "script_timeline_aligned.py"
    json_path = args.output_dir / "alignment_result.json"
    report_path = args.output_dir / "alignment_report.txt"

    try:
        aligned_timeline, audio_schedule, audio_output_path, final_audio_duration = rebuild_audio(
            args.audio,
            timeline,
            line_ranges,
            hypothesis_words,
            mapping,
            args.output_dir,
            args.min_gap,
        )
    except Exception as exc:
        print(f"Erreur pendant la reconstruction de l'audio : {exc}", file=sys.stderr)
        return 1

    write_srt(srt_path, aligned_timeline)
    write_python_timeline(py_path, aligned_timeline)

    transcript_text = str(result.get("text", "")).strip()
    detailed_json = {
        "source_audio": str(args.audio),
        "output_audio": str(audio_output_path),
        "model": args.model,
        "language": result.get("language", args.language),
        "source_audio_duration": round(audio_duration, 3),
        "output_audio_duration": round(final_audio_duration, 3),
        "transcript": transcript_text,
        "timeline": [item for item in aligned_timeline if not item.get("_audio_missing")],
        "audio_schedule": audio_schedule,
        "line_alignment": [
            {
                "index": i,
                "text": timeline[i]["text"],
                "matched_words": line_match_stats[i][0],
                "total_words": line_match_stats[i][1],
                "match_ratio": round(line_match_stats[i][0] / max(1, line_match_stats[i][1]), 3),
                "audio_missing": bool(aligned_timeline[i].get("_audio_missing")),
            }
            for i in range(len(timeline))
        ],
        "whisper_words": hypothesis_words,
    }
    json_path.write_text(json.dumps(detailed_json, ensure_ascii=False, indent=2), encoding="utf-8")

    total_reference_words = len(reference_words)
    total_matched_words = len(mapping)
    report_lines = [
        "RAPPORT D'ALIGNEMENT VOIX OFF / SCRIPT",
        "=" * 42,
        f"Audio source : {args.audio}",
        f"Audio reconstruit : {audio_output_path}",
        f"Modele Whisper : {args.model}",
        f"Duree audio source : {audio_duration:.2f} s",
        f"Duree audio genere : {final_audio_duration:.2f} s",
        f"Mots du script alignes : {total_matched_words}/{total_reference_words} ({100 * total_matched_words / max(1, total_reference_words):.1f} %)",
        "",
        "TRANSCRIPTION WHISPER",
        "---------------------",
        transcript_text,
        "",
        "RECONSTRUCTION AUDIO",
        "-------------------",
        "Les extraits sont places au plus pres des horaires demandes. Si deux extraits se chevauchent, le second est decale pour eviter de superposer les voix.",
        "",
        "DETAIL PAR LIGNE (les lignes sous 70 % meritent une verification)",
        "-----------------------------------------------------------------",
    ]
    for i, item in enumerate(timeline):
        matched, total = line_match_stats[i]
        ratio = matched / max(1, total)
        flag = "  <-- A VERIFIER" if ratio < 0.70 else ""
        scheduled = audio_schedule[i]
        if not scheduled["matched_in_source"]:
            location = "audio non retrouve"
        else:
            location = (
                f"demande {scheduled['requested_start']:.2f}s, sortie {scheduled['output_start']:.2f}s, "
                f"decalage +{scheduled['shift_seconds']:.2f}s"
            )
        report_lines.append(
            f"{i + 1:02d}. {matched}/{total} mots ({ratio:.0%}) | {location} : {item['text']}{flag}"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nTermine. Fichiers crees :")
    print(f"  Audio reconstruit : {audio_output_path}")
    print(f"  Sous-titres SRT   : {srt_path}")
    print(f"  Liste Python      : {py_path}")
    print(f"  Rapport           : {report_path}")
    print(f"  Resultat JSON     : {json_path}")
    print(f"Alignement global : {total_matched_words}/{total_reference_words} mots ({100 * total_matched_words / max(1, total_reference_words):.1f} %)")
    missing_lines = sum(1 for item in aligned_timeline if item.get("_audio_missing"))
    if missing_lines:
        print(f"Attention : {missing_lines} ligne(s) non retrouvee(s) dans l'audio. Consulte alignment_report.txt.")
    print("Note : quand un horaire est trop court, les phrases suivantes sont decalees plutot que coupees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

