# CLAUDE.md

Personal fork of **WanGP** (DeepBeepMeep's local generative-AI super app: video, image, audio/TTS, 3D, worlds, Deepy assistant). Upstream code is not ours; our own work lives in a few personal plugins.

## Repo layout & remotes

- `origin` = `github.com/lsandini/WanGP` (our fork), `upstream` = `github.com/deepbeepmeep/WanGP`.
- `main` tracks upstream. Our work is on feature branches (currently `talking-avatar-photo-editor`).
- **Don't modify upstream files** (`wgp.py`, `shared/`, `models/`, ...) unless unavoidable — it makes upstream merges painful. Put changes in our plugins.
- Our own files:
  - `plugins/wan2gp-talking-avatar/` — Talking Avatar plugin
  - `plugins/wan2gp-photo-editor/` — Photo Editor plugin
  - `start_wangp.bat` + `wangp.ico` — Windows launcher used by the desktop shortcut
  - `wgp_config.backup.json` (before plugins), `wgp_config.backup2.json` (Talking Avatar enabled, before Photo Editor) — config snapshots
  - `.gitignore` exceptions: `/plugins/wan2gp-*/` is ignored upstream, so each of our plugin folders needs a `!/plugins/<name>/` line. **Add one for every new plugin.**
- Ignored, never commit: `ckpts/`, `loras/`, `outputs/`, `settings/`, `workspaces/`, `wgp_config.json`, `env/`.

## Environment

- Windows 10, NVIDIA RTX 5060 Ti (16 GB VRAM). Python 3.11 venv in `env/` (`env\Scripts\python.exe`).
- Run: `start_wangp.bat` → `python wgp.py --multiple-images --advanced --open-browser` (Gradio web UI; exit code 42 = restart loop).
- Plugins are enabled via `enabled_plugins` in `wgp_config.json` (or the Plugins tab in the UI). WanGP must be restarted to pick up plugin code changes.
- No test suite for our plugins; verify by running WanGP and using the tab. Quick syntax check: `env\Scripts\python.exe -m py_compile plugins/<name>/plugin.py`.

## Plugin conventions (both of our plugins follow these)

Reference: `docs/PLUGINS.md` (plugin API, Deepy tools/jobs), `docs/API.md`, `docs/DEEPY.md`.

- Folder = `__init__.py` + `plugin.py` (class subclassing `shared.utils.plugins.WAN2GPPlugin`) + `plugin_info.json` (`"type": "app"`, author Lorenzo Sandini).
- `setup_ui()` calls `self.request_component("state")` and `self.add_tab(...)`; UI is built in `create_config_ui(api_session)`.
- **Job submission**: build settings from WanGP's own Deepy tool templates — `shared.deepy.tool_settings.clone_tool_preset(TOOL, TEMPLATE)` (drop `client_id`, `type`, `model_filename`) or `api_session.get_default_settings(model_type)` — then submit with `self.deepy.web_session().submit_task(settings)` (falls back to `api_session`). Jobs go through WanGP's normal queue, so only one model is loaded at a time.
- **Progress**: poll `job.done`, and show WanGP's own progress bar via `self.deepy._service.generation_view()["html"]` in a `WangpProgress.component()`; `WangpProgress().status(text)` + `.render(active=False)` for static status lines. Event handlers are generators; `show_progress="hidden"`.
- Stop button: set a flag and `job.cancel()`; register with `queue=False`.
- Errors: raise `gr.Error(...)` with a user-readable message.
- Gradio's waveform `gr.Audio` fails to reload new takes during a running event → use a plain HTML `<audio>` served from `/gradio_api/file=<abs path>?v=<mtime>`.

## Talking Avatar (`plugins/wan2gp-talking-avatar/plugin.py`)

Script + portrait → lip-synced talking video.
1. **Speech**: OmniVoice (`omnivoice`) in voice-design mode (no reference audio; voice text in `alt_prompt`, language in `model_mode`; Finnish uses `auto`). Speed control: estimate natural length with OmniVoice's `RuleDurationEstimator` (25 tokens/s), divide by speed, set `duration_seconds` + `custom_settings.adjust_speed_to_max_duration = "Yes"`.
2. **Video**: InfiniteTalk via Deepy tool `gen_video_with_speech`, template `Infinitalk` (lightx2v, 4 steps, 81-frame windows, 25 fps). `audio_guide` = speech file, `image_refs=[photo]`, `video_prompt_type="KI"`, `video_length` = audio length × 25 + 1 s margin.

## Photo Editor (`plugins/wan2gp-photo-editor/plugin.py`)

Instruction-based photo edits via Deepy tool `edit_image` templates: Qwen Image 2.1 Turbo (default, supports painted-area inpainting), Qwen Edit 2511, Flux 2 Klein 9B. With a painted mask, the regenerated area is blended back so all other pixels stay identical. Variants, before/after slider, JPEG/PNG output, detail level (≈1 MP or up to 2.4 MP).

## User preferences

- User writes Italian, Finnish and English content; keep language options for those.
- Commit only when asked; commit messages end with the Claude co-author line.
