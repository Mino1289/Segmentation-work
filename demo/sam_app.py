#!/usr/bin/env python3
"""Interactive SAM demo with Gradio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from starlette import status

# Gradio still reads the deprecated Starlette 422 constant at request time.
status.HTTP_422_UNPROCESSABLE_ENTITY = status.HTTP_422_UNPROCESSABLE_CONTENT

import gradio as gr
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.sam_inference import (
    DEFAULT_CHECKPOINT,
    SAMPredictor,
    mask_to_binary_image,
    mask_to_color_overlay,
    render_candidate_gallery,
)
from eval.data import get_ade20k_sample, get_ade20k_sample_count

_predictor: Optional[SAMPredictor] = None
_current_image: Optional[Image.Image] = None
_points: List[Tuple[float, float]] = []
_labels: List[int] = []


def get_predictor(checkpoint: str) -> SAMPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SAMPredictor(checkpoint_path=checkpoint)
    return _predictor


def reset_points() -> tuple:
    global _points, _labels
    _points = []
    _labels = []
    return [], "Points effacés."


def load_test_image(index: int) -> tuple:
    global _current_image
    image, _ = get_ade20k_sample(index, split="test")
    _current_image = image
    reset_points()
    return image, f"Image test ADE20K index {index}"


def load_uploaded_image(uploaded) -> tuple:
    global _current_image
    if uploaded is None:
        return None, "Aucune image uploadée."
    if isinstance(uploaded, Image.Image):
        image = uploaded.convert("RGB")
    else:
        image = Image.fromarray(uploaded).convert("RGB")
    _current_image = image
    reset_points()
    return image, "Image personnelle chargée."


def handle_select(
    evt: gr.SelectData,
    point_mode: str,
    checkpoint: str,
) -> tuple:
    global _points, _labels, _current_image

    if _current_image is None:
        return None, None, [], "Chargez d'abord une image."

    x, y = evt.index[0], evt.index[1]
    label = 1 if point_mode == "foreground" else 0
    _points.append((float(x), float(y)))
    _labels.append(label)

    if not _points:
        return None, None, [], "Ajoutez au moins un point."

    try:
        predictor = get_predictor(checkpoint)
        result = predictor.predict(_current_image, _points, _labels)
        overlay = mask_to_color_overlay(_current_image, result["best_mask"])
        binary = mask_to_binary_image(result["best_mask"])
        gallery = render_candidate_gallery(
            _current_image, result["all_masks"], result["all_ious"]
        )
        status = (
            f"{len(_points)} point(s) — meilleur masque IoU={result['best_iou']:.3f}"
        )
        return overlay, binary, gallery, status
    except FileNotFoundError as exc:
        return None, None, [], str(exc)
    except Exception as exc:
        return None, None, [], f"Erreur: {exc}"


def clear_and_refresh(checkpoint: str) -> tuple:
    reset_points()
    return None, None, [], "Points effacés. Cliquez sur l'image pour segmenter."


def build_demo(checkpoint: str, share: bool) -> gr.Blocks:
    test_count = get_ade20k_sample_count("test")

    with gr.Blocks(title="SAM Demo — ADE20K") as demo:
        gr.Markdown(
            "# Démo interactive SAM (ViT-H)\n"
            "Chargez une image du **test ADE20K** ou uploadez la vôtre, "
            "puis cliquez sur l'image pour placer des points de prompt."
        )

        checkpoint_state = gr.State(checkpoint)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Source image")
                test_index = gr.Slider(
                    minimum=0,
                    maximum=max(test_count - 1, 0),
                    value=0,
                    step=1,
                    label="Index test ADE20K",
                )
                btn_load_test = gr.Button("Charger image test")
                upload = gr.Image(type="pil", label="Ou uploader une image personnelle")
                point_mode = gr.Radio(
                    choices=["foreground", "background"],
                    value="foreground",
                    label="Type de point",
                    info="foreground = objet à segmenter, background = zone à exclure",
                )
                btn_clear = gr.Button("Effacer les points")
                status = gr.Textbox(label="Statut", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Cliquez sur l'image pour segmenter")
                input_image = gr.Image(type="pil", label="Image interactive", interactive=True)
                with gr.Row():
                    output_overlay = gr.Image(label="Overlay (meilleur masque)")
                    output_mask = gr.Image(label="Masque binaire")
                gallery = gr.Gallery(label="Masques candidats (multimask)", columns=3)

        btn_load_test.click(
            fn=load_test_image,
            inputs=[test_index],
            outputs=[input_image, status],
        )
        upload.change(
            fn=load_uploaded_image,
            inputs=[upload],
            outputs=[input_image, status],
        )
        input_image.select(
            fn=handle_select,
            inputs=[point_mode, checkpoint_state],
            outputs=[output_overlay, output_mask, gallery, status],
        )
        btn_clear.click(
            fn=clear_and_refresh,
            inputs=[checkpoint_state],
            outputs=[output_overlay, output_mask, gallery, status],
        )

        demo.load(
            fn=load_test_image,
            inputs=[test_index],
            outputs=[input_image, status],
        )

    demo.launch(share=share)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM Gradio demo")
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to SAM ViT-H checkpoint",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_demo(checkpoint=args.sam_checkpoint, share=args.share)


if __name__ == "__main__":
    main()
