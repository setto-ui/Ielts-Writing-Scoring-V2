"""
test_scorer.py - Interactive CLI tester & AI Tutor for IELTS Writing Scorer v2.2
Powered by Tri-Specialist Architecture, Rich Terminal UI, and Diagnostic Tutor Engine.

Usage:
    python test_scorer.py
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path
import numpy as np

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

console = Console()

console.print("[dim]Loading IELTS scoring models and diagnostic engines, please wait...[/dim]", highlight=False)

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.two_stage_model import TriSpecialistModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate
from scorer.tutor_engine import diagnose_errors, generate_band_8_rewrite

MODEL_PATH_TRI = "data/model_tri_specialist.pkl"
MODEL_PATH_BASE = "data/model_exp_6JB.pkl"

BAND_LABEL = {
    9.0: "Expert User",
    8.5: "Very Good - Expert",
    8.0: "Very Good User",
    7.5: "Good - Very Good",
    7.0: "Good User",
    6.5: "Competent - Good",
    6.0: "Competent User",
    5.5: "Modest - Competent",
    5.0: "Modest User",
    4.5: "Limited User",
    4.0: "Limited User",
    3.5: "Extremely Limited User",
    3.0: "Extremely Limited User",
}


def make_bar(score: float, width: int = 20) -> str:
    """Create a colored bar representing the band score."""
    pct = min(1.0, max(0.0, score / 9.0))
    filled = int(round(pct * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def collect_multiline_input(step_title: str, prompt_hint: str) -> str:
    """Prompt user for multiline text until they enter 'end'."""
    console.print()
    console.print(Panel(
        f"[bold cyan]{step_title}[/bold cyan]\n"
        f"[dim]{prompt_hint}[/dim]\n"
        f"[yellow]>> Type 'end' on a new line and press Enter when finished <<[/yellow]",
        border_style="cyan"
    ))
    
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().lower() == "end":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_scoring_and_tutoring(prompt: str, essay: str) -> None:
    """Run full scoring pipeline, error diagnosis, and model rewrite."""
    with console.status("[bold green]Analyzing essay with Tri-Specialist Model & AI Tutor Engine...[/bold green]"):
        # Load model
        if Path(MODEL_PATH_TRI).exists():
            model = TriSpecialistModel.load(MODEL_PATH_TRI)
            is_tri = True
            model_name = "Tri-Specialist Two-Stage v2.2"
        else:
            model = IELTSModel.load(MODEL_PATH_BASE)
            is_tri = False
            model_name = "Baseline LightGBM 6J-B"

        # 1. Feature extraction
        raw = extract_all_features(essay, prompt)
        norm = normalize_features(raw)

        # 2. ML Prediction
        X = features_to_array(norm, FEATURE_KEYS).reshape(1, -1)
        ml_raw = model.predict(X)
        ml_preds = {
            k: float(v[0]) if hasattr(v, "__len__") else float(v)
            for k, v in ml_raw.items()
        }

        # 3. Constraint checking
        score_result = score_essay(
            normalized_features=norm,
            raw_features=raw,
            ml_predictions=ml_preds,
        )
        constrained, rules = apply_constraints(score_result["scores"], raw)

        # 4. Band rounding
        if is_tri:
            overall = round_to_ielts_band(float(constrained.get("overall", ml_preds["overall"])))
        else:
            overall = calibrate(float(np.mean(list(constrained.values()))))
        rounded = {k: round_to_ielts_band(v) for k, v in constrained.items()}

        # 5. Diagnostic & Error Analysis
        diagnosis = diagnose_errors(essay, prompt)

        # 6. Generate Band 8.0+ Rewrite
        rewrite_result = generate_band_8_rewrite(essay, prompt, diagnosis)

    # ─── Render Rich Output ────────────────────────────────────

    # --- 1. OVERALL SCORECARD ---
    wc = int(raw.get("word_count", len(essay.split())))
    wc_status = "[bold green]PASS (≥ 250 words)[/bold green]" if wc >= 250 else "[bold red]PENALTY (< 250 words)[/bold red]"
    label = BAND_LABEL.get(overall, "User")
    bar_str = make_bar(overall, width=24)

    # Confidence calculation
    if overall <= 5.5:
        conf_level = "[bold yellow]MODERATE (±0.5 - 1.0)[/bold yellow]"
        conf_range = f"Band {max(4.0, overall - 1.0):.1f} - {min(9.0, overall + 1.0):.1f}"
    elif overall <= 7.0:
        conf_level = "[bold green]HIGH (86.3% within ±0.5)[/bold green]"
        conf_range = f"Band {max(4.0, overall - 0.5):.1f} - {min(9.0, overall + 0.5):.1f}"
    else:
        conf_level = "[bold cyan]GOOD (±0.5 - 1.0)[/bold cyan]"
        conf_range = f"Band {max(4.0, overall - 1.0):.1f} - {min(9.0, overall + 1.0):.1f}"

    score_text = Text()
    score_text.append(f"\n  OVERALL BAND SCORE: ", style="bold white")
    score_text.append(f"{overall:.1f} ", style="bold green" if overall >= 6.5 else "bold yellow")
    score_text.append(f"({label})\n", style="italic cyan")
    score_text.append(f"  Score Gauge: [{bar_str}] {overall:.1f} / 9.0\n\n", style="bold green")
    score_text.append(f"  • Model Architecture : {model_name}\n", style="white")
    score_text.append(f"  • Word Count         : {wc} words  ({wc_status})\n", style="white")
    score_text.append(f"  • Topic Coverage     : {diagnosis['topic_coverage_pct']}% prompt keywords covered\n", style="white")
    score_text.append(f"  • Statistical Range  : {conf_range} (Confidence: {conf_level})\n", style="white")

    console.print(Panel(
        score_text,
        title="[bold blue]🏆 IELTS OFFICIAL SCORECARD[/bold blue]",
        subtitle="[dim]Calibrated against Cambridge IELTS Barem[/dim]",
        border_style="bright_blue",
        padding=(1, 2)
    ))

    # --- 2. CRITERIA BREAKDOWN TABLE ---
    table = Table(title="[bold cyan]📊 4 ASSESSMENT CRITERIA BREAKDOWN[/bold cyan]", border_style="cyan", header_style="bold cyan")
    table.add_column("Criterion", style="white", width=28)
    table.add_column("Band", justify="center", style="bold magenta", width=8)
    table.add_column("Visual Scale", style="green", width=22)
    table.add_column("Key Diagnostic Indicators", style="dim white")

    criteria_meta = [
        ("task_response", "Task Response (TR)", f"Coverage: {diagnosis['topic_coverage_pct']}% | Words: {wc}"),
        ("coherence",     "Coherence & Cohesion (CC)", f"Paragraphs: {diagnosis['num_paragraphs']} | Markers: {len(diagnosis['found_discourse_markers'])} detected"),
        ("lexical",       "Lexical Resource (LR)", f"Overused terms: {len(diagnosis['overused_words'])} | Diversity MATTR: {raw.get('mattr', 0):.2f}"),
        ("grammar",       "Grammatical Range (GRA)", f"Complex clauses: {diagnosis['complex_ratio']}% | Passives: {diagnosis['passive_ratio']}%"),
    ]

    for key, name, ind in criteria_meta:
        b = rounded.get(key, overall)
        table.add_row(name, f"{b:.1f}", make_bar(b, width=18), ind)

    console.print(table)
    console.print()

    # --- 3. ERROR & WEAKNESS DIAGNOSIS PANEL ---
    err_text = Text()

    # Grammar issues
    err_text.append("🔴 1. Cú pháp & Cấu trúc câu (Grammar & Sentence Structure):\n", style="bold red")
    if diagnosis["run_on_sentences"]:
        for r in diagnosis["run_on_sentences"]:
            err_text.append(f"   • [Câu quá dài - {r['length']} từ]: ", style="bold yellow")
            err_text.append(f"\"{r['sentence'][:90]}...\"\n", style="italic white")
            err_text.append(f"     ↳ Gợi ý: {r['suggestion']}\n", style="dim cyan")
    else:
        err_text.append("   ✓ Không phát hiện câu quá dài (run-on sentences).\n", style="green")

    if diagnosis["choppy_sentences"]:
        for c in diagnosis["choppy_sentences"]:
            err_text.append(f"   • [Câu quá ngắn/rời rạc - {c['length']} từ]: ", style="bold yellow")
            err_text.append(f"\"{c['sentence']}\"\n", style="italic white")
            err_text.append(f"     ↳ Gợi ý: {c['suggestion']}\n", style="dim cyan")

    # Lexical repetition
    err_text.append("\n🟡 2. Lỗi Lặp từ & Đề xuất Từ vựng Học thuật (Band 8.0+ Lexical Upgrades):\n", style="bold yellow")
    if diagnosis["overused_words"]:
        for item in diagnosis["overused_words"]:
            w = item["word"]
            cnt = item["count"]
            err_text.append(f"   • Từ '[bold white]{w}[/bold white]' bị lặp [bold red]{cnt} lần[/bold red].\n")
            if item["upgrade"]:
                alts = ", ".join(item["upgrade"]["band_8"])
                colloc = item["upgrade"]["collocation"]
                err_text.append(f"     ↳ [cyan]Thay thế nâng cao:[/cyan] {alts}\n")
                err_text.append(f"     ↳ [green]Collocation gợi ý:[/green] [italic]{colloc}[/italic]\n")
            else:
                err_text.append(f"     ↳ [dim]Nên dùng đại từ thay thế hoặc paraphrase bằng danh từ tương đương.[/dim]\n")
    else:
        err_text.append("   ✓ Từ vựng phân bổ tự nhiên, không có từ nào bị lặp quá mức.\n", style="green")

    # Coherence & Task
    err_text.append("\n🔵 3. Bố cục & Mạch lạc (Coherence & Cohesion):\n", style="bold blue")
    if diagnosis["coherence_warnings"]:
        for w in diagnosis["coherence_warnings"]:
            err_text.append(f"   • [Cảnh báo]: {w}\n", style="yellow")
    else:
        err_text.append("   ✓ Cấu trúc phân đoạn và số lượng từ nối đạt yêu cầu.\n", style="green")

    if diagnosis["missing_keywords"]:
        missing_str = ", ".join(diagnosis["missing_keywords"])
        err_text.append(f"\n🟣 4. Độ bám sát đề (Task Coverage Gaps):\n", style="bold magenta")
        err_text.append(f"   • Các khía cạnh/từ khóa chưa được khai thác sâu: [italic]{missing_str}[/italic]\n", style="white")

    console.print(Panel(
        err_text,
        title="[bold red]⚠️ CHI TIẾT LỖI SAI & ĐIỂM YẾU CỦA BÀI VIẾT[/bold red]",
        border_style="red",
        padding=(1, 2)
    ))

    # --- 4. ACTION PLAN ---
    action_text = Text()
    action_text.append("🎯 3 Hành động cụ thể để tăng +1.0 Band trong lần viết tới:\n\n", style="bold green")
    action_text.append("1. [bold white]Đa dạng hóa cấu trúc câu:[/bold white] Tăng cường dùng mệnh đề quan hệ (Which, Who) và mệnh đề trạng ngữ chỉ sự nhượng bộ (Whereas, Although) thay vì viết nhiều câu đơn nối bằng 'and'/'so'.\n\n", style="white")
    action_text.append("2. [bold white]Nâng cấp Academic Collocations:[/bold white] Thay thế các từ vựng thông dụng ('people', 'good', 'problem') bằng các cụm học thuật cố định như đã gợi ý ở bảng trên.\n\n", style="white")
    action_text.append("3. [bold white]Bám sát toàn diện đề bài:[/bold white] Đảm bảo giải quyết đủ 100% các vế hỏi của đề trong cả hai đoạn thân bài (Body paragraphs), mỗi đoạn phát triển trọn vẹn 1-2 luận điểm kèm ví dụ cụ thể.\n", style="white")

    console.print(Panel(
        action_text,
        title="[bold green]🚀 KẾ HOẠCH HÀNH ĐỘNG CẢI THIỆN ĐIỂM SỐ[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))

    # --- 5. BAND 8.0+ MODEL ESSAY REWRITE ---
    model_text = Text()
    model_text.append(f"[bold cyan]Dưới đây là phiên bản viết lại đạt chuẩn Band 8.0+, giữ trọn vẹn luận điểm của bạn nhưng nâng cấp từ vựng học thuật và liên kết câu:[/bold cyan]\n\n", style="white")
    model_text.append(rewrite_result["model_essay"], style="italic bright_white")
    model_text.append("\n\n" + "─" * 60 + "\n", style="dim")
    model_text.append("💡 [bold yellow]Những điểm nâng cấp cốt lõi trong bài viết mẫu này:[/bold yellow]\n", style="white")
    for up in rewrite_result["key_upgrades"]:
        model_text.append(f"  • {up}\n", style="dim cyan")

    console.print(Panel(
        model_text,
        title="[bold magenta]✍️ BÀI VIẾT THAM KHẢO CHUẨN BAND 8.0+ (MODEL REWRITE)[/bold magenta]",
        subtitle="[dim]Academic Reference Essay for Study[/dim]",
        border_style="magenta",
        padding=(1, 2)
    ))
    console.print()


def main():
    console.print()
    console.print(Panel(
        "[bold cyan]IELTS Writing Examiner & AI Tutor v2.2[/bold cyan]\n"
        "[white]Automatic Band Scoring • Deep Diagnostic Analysis • Band 8.0+ Model Rewrite[/white]\n"
        "[dim]100% Offline Local Pipeline | Powered by Tri-Specialist & Rich Terminal[/dim]",
        border_style="bright_blue",
        padding=(1, 2)
    ))

    while True:
        prompt = collect_multiline_input(
            "[1/2] EXAM PROMPT",
            "Nhập đề bài IELTS Writing Task 2 (ví dụ: 'Some people believe that... Do you agree or disagree?')"
        )
        if not prompt:
            console.print("[red][!] Đề bài trống. Vui lòng nhập lại.[/red]")
            continue

        essay = collect_multiline_input(
            "[2/2] STUDENT ESSAY",
            "Dán bài viết của học sinh (tối thiểu 100 từ, chuẩn IELTS từ 250 từ trở lên)"
        )
        if not essay:
            console.print("[red][!] Bài viết trống. Vui lòng nhập lại.[/red]")
            continue

        wc = len(essay.split())
        if wc < 50:
            console.print(f"[red][!] Bài viết quá ngắn ({wc} từ). IELTS cần tối thiểu 250 từ để chấm chính xác.[/red]\n")
            continue

        run_scoring_and_tutoring(prompt, essay)

        console.print("[bold yellow]Nhấn Enter để tiếp tục chấm bài khác, hoặc gõ 'exit' để thoát:[/bold yellow] ", end="")
        choice = input().strip().lower()
        if choice in ("exit", "quit", "q"):
            console.print("\n[bold green]Cảm ơn bạn đã sử dụng IELTS Writing Scorer & Tutor. Tạm biệt![/bold green]\n")
            break


if __name__ == "__main__":
    main()
