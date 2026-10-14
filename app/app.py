"""
app/app.py
Gradio demo for DocRescue.

Features:
- Upload a phone-photo of a document
- Pipeline: perspective correction → gain-map U-Net shadow removal
- Side-by-side before/after display
- OCR text extraction + comparison panel
- Works on CPU (no GPU required for demo)

Run locally:
    python app/app.py --ckpt experiments/checkpoints/best.pth

Share publicly (Colab/HF Spaces):
    python app/app.py --ckpt best.pth --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import gradio as gr

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import correct_perspective
from src.infer import gain_map_rescue, load_checkpoint
from src.classical import classical_pipeline
from src.ocr import run_ocr, normalize_text
from src.models import build_model


# ---------------------------------------------------------------------------
# Global state (model loaded once)
# ---------------------------------------------------------------------------

_model = None
_device = None
_cfg = None


def _load_model_once(ckpt_path: str, cfg: dict) -> None:
    global _model, _device, _cfg
    _cfg = cfg
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(cfg).to(_device)
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(_model, ckpt_path, device=_device)
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    else:
        print("⚠ No checkpoint found; running with random weights (demo mode).")
    _model.eval()


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_image(
    input_image: np.ndarray,
    use_perspective: bool,
    use_unet: bool,
    use_classical_fallback: bool,
    ocr_engine: str,
) -> tuple:
    """Main processing function called by Gradio.

    Args:
        input_image:           RGB uint8 array from Gradio.
        use_perspective:       Apply perspective correction.
        use_unet:              Apply U-Net gain-map shadow removal.
        use_classical_fallback: Apply classical baseline if U-Net disabled.
        ocr_engine:            "tesseract" or "easyocr".

    Returns:
        (corrected_image, ocr_original, ocr_corrected, status_message)
    """
    if input_image is None:
        return None, "", "", "Please upload an image."

    result = input_image.copy()
    steps_log = []

    # Stage 1: Perspective correction
    if use_perspective:
        result, success = correct_perspective(result)
        steps_log.append("✓ Perspective correction" if success else "⚠ Perspective correction skipped (no quad found)")
    
    # Stage 2: Illumination removal
    if use_unet and _model is not None:
        result = gain_map_rescue(
            result,
            _model,
            device=_device,
            input_size=_cfg["preprocessing"]["image_size"],
            blur_ksize=_cfg["inference"]["gain_map_blur_ksize"],
            gain_min_clamp=_cfg["inference"]["gain_map_min_clamp"],
        )
        steps_log.append("✓ U-Net gain-map shadow removal")
    elif use_classical_fallback:
        result = classical_pipeline(result)
        steps_log.append("✓ Classical illumination baseline (background division + CLAHE)")

    # Stage 3: OCR
    ocr_original = run_ocr(input_image, engine=ocr_engine)
    ocr_corrected = run_ocr(result, engine=ocr_engine)

    status = "\n".join(steps_log) if steps_log else "No processing applied."
    return result, ocr_original, ocr_corrected, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(ckpt_path: str, cfg: dict) -> gr.Blocks:
    _load_model_once(ckpt_path, cfg)

    with gr.Blocks(
        title="DocRescue – Document Enhancement",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="""
        .title-block { text-align: center; margin-bottom: 1rem; }
        .result-panel { border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="title-block">
            <h1>📄 DocRescue</h1>
            <p>Shadow removal + perspective correction for phone-scanned documents, judged by OCR accuracy.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Upload Document Photo", type="numpy")

                gr.Markdown("### Pipeline Options")
                use_perspective = gr.Checkbox(label="Perspective Correction (OpenCV)", value=True)
                use_unet = gr.Checkbox(
                    label="U-Net Shadow Removal (gain-map)",
                    value=_model is not None,
                )
                use_classical = gr.Checkbox(
                    label="Classical Baseline (fallback if U-Net off)",
                    value=False,
                )
                ocr_engine = gr.Radio(
                    choices=["tesseract", "easyocr"],
                    value="tesseract",
                    label="OCR Engine",
                )

                run_btn = gr.Button("🚀 Process", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    output_img = gr.Image(label="Corrected Document", type="numpy")
                status_box = gr.Textbox(label="Processing Steps", lines=4, interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### OCR – Original")
                        ocr_orig = gr.Textbox(label="", lines=6, interactive=False)
                    with gr.Column():
                        gr.Markdown("### OCR – Corrected")
                        ocr_corr = gr.Textbox(label="", lines=6, interactive=False)

        # Examples (illustrative paths; swap with real images)
        gr.Examples(
            examples=[],
            inputs=[input_img],
        )

        run_btn.click(
            fn=process_image,
            inputs=[input_img, use_perspective, use_unet, use_classical, ocr_engine],
            outputs=[output_img, ocr_orig, ocr_corr, status_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="DocRescue Gradio Demo")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ckpt", type=str, default="experiments/checkpoints/best.pth")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    demo = build_ui(args.ckpt, cfg)
    demo.launch(share=args.share, server_port=args.port)
