"""Photo Editor: fine, instruction-based edits of a photo (remove a blemish, change a detail...).

Paint over the area to change (optional) and describe the edit.
- With a painted area, only that area is regenerated (Qwen Image 2.1 native inpainting), and the
  result is blended back into the untouched original, so every other pixel stays identical.
- Without a painted area, the whole photo is edited from the instruction.
Jobs go through WanGP's normal queue, using WanGP's own Deepy "edit_image" templates.
"""
import math
import os
import time

import gradio as gr
from PIL import Image, ImageChops, ImageFilter

from shared.gradio.progress import WangpProgress
from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Photo Editor"
PlugIn_Id = "PhotoEditor"

EDIT_TOOL = "edit_image"
# label -> Deepy edit_image template
MODELS = {
    "Qwen Image 2.1 Turbo 7B (fast, supports painted areas)": "Qwen Image 2.1 Turbo 8 Steps",
    "Qwen Edit 2511 20B (slower, strongest identity keeping)": "Qwen Edit 2511 lighx2v 4 steps",
    "Flux 2 Klein 9B (fast alternative)": "Flux Klein 9B",
}
DEFAULT_MODEL = next(iter(MODELS))
# label -> (file extension, PIL save options)
OUTPUT_FORMATS = {
    "JPEG (high quality, smaller file)": (".jpg", {"quality": 95, "subsampling": 0}),
    "PNG (lossless, larger file)": (".png", {}),
}
DEFAULT_OUTPUT_FORMAT = next(iter(OUTPUT_FORMATS))
DETAIL_LEVELS = {
    "High (close to original size, up to 2.4 MP)": 2.4e6,
    "Fast (about 1 MP)": 1.05e6,
}
DEFAULT_DETAIL = next(iter(DETAIL_LEVELS))
SIZE_STEP = 32  # Qwen 2.1 VAE block size; also fine for the other models
DEFAULT_INSTRUCTION = (
    "Remove the small band-aid under the eye, revealing natural clean skin that matches the surrounding "
    "skin texture, tone and lighting. Keep everything else exactly the same."
)
EDITOR_HEIGHT = 720


def _generation_size(width, height, max_pixels):
    """Same aspect ratio as the photo, at most max_pixels, at least ~1 MP, multiples of SIZE_STEP."""
    pixels = width * height
    target = min(max(pixels, 1.05e6), max_pixels)
    scale = math.sqrt(target / pixels)
    w = max(SIZE_STEP, int(round(width * scale / SIZE_STEP)) * SIZE_STEP)
    h = max(SIZE_STEP, int(round(height * scale / SIZE_STEP)) * SIZE_STEP)
    return w, h


def _mask_from_editor(value):
    """White where the user painted (any layer), black elsewhere. None if nothing was painted."""
    layers = (value or {}).get("layers") or []
    mask = None
    for layer in layers:
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            layer = Image.open(layer)
        if layer.mode in ("RGBA", "LA"):
            painted = layer.getchannel("A")
        else:
            # RGB layers (editor image_mode="RGB"): unpainted pixels are black, brush strokes are coloured.
            painted = layer.convert("L")
        mask = painted if mask is None else ImageChops.lighter(mask, painted)
    if mask is None:
        return None
    mask = mask.point(lambda v: 255 if v > 20 else 0)
    return mask if mask.getbbox() else None


def _soft_mask(mask, grow, feather):
    if grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * int(grow) + 1))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))
    return mask


def _status_html(text):
    """A finished-style WanGP progress bar showing a plain status line."""
    from shared.gradio.progress import WangpProgress

    bar = WangpProgress()
    bar.status(text)
    return bar.render(active=False)


def _relay(job_updates, wrap):
    """yield from a run_job generator, wrapping each progress HTML; returns run_job's return value."""
    while True:
        try:
            html = next(job_updates)
        except StopIteration as done:
            return done.value
        yield wrap(html)


class PhotoEditorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = "Instruction-based photo edits, optionally limited to a painted area."

    def setup_ui(self):
        # The WebUI-backed api_session needs the live Gradio state to reach WanGP's queue.
        self.request_component("state")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state):
        return None

    def on_tab_deselect(self, state):
        return None

    @staticmethod
    def _edit_settings(api_session, model_label, instruction, photo_path, mask_path, size, seed, variants):
        from shared.deepy import tool_settings

        template = tool_settings.clone_tool_preset(EDIT_TOOL, MODELS.get(model_label, MODELS[DEFAULT_MODEL]))
        for key in ("client_id", "type", "model_filename"):
            template.pop(key, None)
        settings = api_session.get_default_settings(template["model_type"])
        settings.update(template)
        settings.update(
            {
                "prompt": instruction.strip(),
                "resolution": f"{size[0]}x{size[1]}",
                "seed": int(seed),
                "repeat_generation": int(variants),
                "batch_size": 1,
                "prompt_enhancer": "",
            }
        )
        for key in ("image_start", "image_end", "image_refs", "image_guide", "image_mask"):
            settings.pop(key, None)
        if mask_path and template["model_type"].startswith("qwen_image_21"):
            # Native inpainting: the model sees the whole photo but only regenerates the painted area.
            settings.update(
                {
                    "image_mode": 2,
                    "video_prompt_type": "VAG",
                    "image_guide": photo_path,
                    "image_mask": mask_path,
                    "model_mode": 0,  # Masked Denoising
                    "denoising_strength": 1.0,
                }
            )
        else:
            settings.update({"image_mode": 1, "video_prompt_type": "KI", "image_refs": [photo_path]})
        return settings

    def _queue_session(self, api_session):
        """Server-side session that joins WanGP's queue directly (no browser round-trip); falls back to the UI bridge."""
        try:
            return self.deepy.web_session()
        except Exception:
            return api_session

    def create_config_ui(self, api_session):
        active = {"job": None, "stop": False}

        def live_progress_html():
            """The same progress bar the Media Generator shows (downloads, model loading, denoising steps)."""
            try:
                return self.deepy._service.generation_view().get("html")
            except Exception:
                return None

        def run_job(settings):
            """Generator: yields WanGP's progress-bar HTML while the job runs, returns the generated files."""
            job = self._queue_session(api_session).submit_task(settings)
            active["job"] = job
            last = None
            try:
                while not job.done:
                    html = live_progress_html()
                    if html and html != last:
                        last = html
                        yield html
                    time.sleep(0.3)
                result = job.result()
            finally:
                active["job"] = None
            if result.cancelled or active["stop"]:
                raise gr.Error("Stopped.")
            if not result.success:
                errors = list(result.errors or [])
                raise gr.Error(f"Edit failed: {errors[0].message if errors else 'no output'}")
            return [str(f) for f in (result.generated_files or []) if f]

        def edit(editor_value, instruction, model_label, detail, grow, feather, seed, variants, output_format):
            """Generator: yields (progress_html, results, compare, selected, last, finals_state)."""
            keep = (gr.update(),) * 5
            photo = (editor_value or {}).get("background")
            if photo is None:
                raise gr.Error("Upload a photo first.")
            if not instruction or not instruction.strip():
                raise gr.Error("Describe the edit first.")
            active["stop"] = False
            if not isinstance(photo, Image.Image):
                photo = Image.open(photo)
            photo = photo.convert("RGB")
            mask = _mask_from_editor(editor_value)
            if mask is not None and mask.size != photo.size:
                mask = mask.resize(photo.size, Image.NEAREST)

            work_dir = os.path.join(os.getcwd(), "outputs", "photo_editor")
            os.makedirs(work_dir, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
            photo_path = os.path.join(work_dir, f"{stamp}_source.png")
            photo.save(photo_path)
            mask_path = None
            if mask is not None:
                # Give the model a little margin around the painted area.
                mask_path = os.path.join(work_dir, f"{stamp}_mask.png")
                _soft_mask(mask, max(2, int(grow) // 2), 0).convert("RGB").save(mask_path)

            size = _generation_size(photo.width, photo.height, DETAIL_LEVELS.get(detail, DETAIL_LEVELS[DEFAULT_DETAIL]))
            yield (_status_html("Queued in WanGP..."),) + keep
            settings = self._edit_settings(self._queue_session(api_session), model_label, instruction, photo_path, mask_path, size, seed, variants)
            files = yield from _relay(run_job(settings), lambda html: (html,) + keep)
            images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            if not images:
                raise gr.Error(f"The model returned no image ({files}).")

            blend = _soft_mask(mask, grow, feather) if mask is not None else None
            finals = []
            for index, path in enumerate(images, 1):
                edited = Image.open(path).convert("RGB").resize(photo.size, Image.LANCZOS)
                if blend is not None:
                    edited = Image.composite(edited, photo, blend)  # untouched pixels outside the painted area
                extension, save_options = OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS[DEFAULT_OUTPUT_FORMAT])
                out = os.path.join(work_dir, f"{stamp}_edit{index}{extension}")
                edited.save(out, **save_options)
                finals.append(out)
            where = "only the painted area was changed" if blend is not None else "the whole photo was edited"
            message = f"Done: {len(finals)} result(s), {where}. Saved in outputs/photo_editor."
            gr.Info(message)
            pair = (photo_path, finals[0])
            yield _status_html(message), finals, pair, finals[0], pair, finals

        def stop():
            active["stop"] = True
            job = active.get("job")
            if job is not None and not job.done:
                job.cancel()

        def pick(finals, pair, evt: gr.SelectData):
            if not finals or not pair or evt.index is None or evt.index >= len(finals):
                return gr.update(), gr.update()
            return finals[evt.index], (pair[0], finals[evt.index])

        def use_as_input(selected):
            if not selected:
                raise gr.Error("Run an edit first (and click the result you want to keep).")
            return selected

        with gr.Column():
            gr.Markdown(
                "### Photo Editor\n"
                "Upload a photo, describe the change, and optionally **paint over the area** to change with the brush.\n"
                "**With a painted area**, only that area is regenerated and everything else stays pixel-identical (best for fine edits like removing a blemish). "
                "**Without**, the whole photo is edited from the description.\n\n"
                "Tip: paint generously around the object (a bit of surrounding skin too). Try 2-4 variants and keep the best; "
                "use **Continue editing from this result** to chain edits."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    editor = gr.ImageEditor(
                        label="Photo (paint over the area to change, optional)",
                        type="pil",
                        image_mode="RGB",
                        sources=["upload", "clipboard"],
                        layers=False,
                        transforms=(),
                        format="png",
                        brush=gr.Brush(colors=["#FF3030"], color_mode="fixed", default_size=40),
                        height=EDITOR_HEIGHT,
                    )
                with gr.Column(scale=2):
                    instruction = gr.Textbox(label="What should change?", value=DEFAULT_INSTRUCTION, lines=5)
                    model = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Image model")
                    detail = gr.Dropdown(list(DETAIL_LEVELS), value=DEFAULT_DETAIL, label="Detail")
                    output_format = gr.Dropdown(list(OUTPUT_FORMATS), value=DEFAULT_OUTPUT_FORMAT, label="Save result as")
                    with gr.Row():
                        variants = gr.Slider(1, 4, value=2, step=1, label="Variants")
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    with gr.Accordion("Blending of the painted area", open=False):
                        grow = gr.Slider(0, 40, value=10, step=1, label="Grow painted area (pixels)")
                        feather = gr.Slider(0, 30, value=8, step=1, label="Soft edge (pixels)")
                    with gr.Row():
                        edit_btn = gr.Button("Edit photo", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    status = WangpProgress.component(visible=True)
            gr.Markdown("#### Result")
            with gr.Row():
                compare = gr.ImageSlider(label="Before / after (drag the slider)", type="filepath", height=EDITOR_HEIGHT, scale=3)
                with gr.Column(scale=2):
                    results = gr.Gallery(label="Variants (click one to compare / keep)", type="filepath", columns=2, height=360, allow_preview=False)
                    selected = gr.Textbox(label="Selected result", interactive=False)
                    reuse_btn = gr.Button("Continue editing from this result")

        last = gr.State(None)
        finals_state = gr.State([])

        # WanGP's own progress bar (status) sits right under the buttons and is refreshed while the job runs.
        edit_btn.click(
            fn=edit,
            inputs=[editor, instruction, model, detail, grow, feather, seed, variants, output_format],
            outputs=[status, results, compare, selected, last, finals_state],
            show_progress="hidden",
        )
        stop_btn.click(fn=stop, queue=False)
        results.select(fn=pick, inputs=[finals_state, last], outputs=[selected, compare])
        reuse_btn.click(fn=use_as_input, inputs=[selected], outputs=[editor])
