from __future__ import annotations

import os
import ssl
import subprocess
import urllib.request


HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_hand_landmarker_task(model_path: str, *, url: str = HAND_LANDMARKER_TASK_URL, timeout_s: int = 30) -> str:
    """
    Ensure `hand_landmarker.task` exists at `model_path`.

    If missing, attempts to download from the official MediaPipe model bucket.
    """

    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # 1) Try Python download first (fast, no external deps).
    try:
        # Some macOS Python builds (notably from python.org) can have missing SSL root certificates,
        # resulting in CERTIFICATE_VERIFY_FAILED. Prefer certifi if available.
        try:
            import certifi  # type: ignore

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()

        with urllib.request.urlopen(url, context=ctx, timeout=timeout_s) as r, open(model_path, "wb") as f:
            f.write(r.read())
    except Exception as e:
        # Clean up partial downloads (common if interrupted).
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception:
            pass

        # 2) Fallback to curl. This often succeeds even when Python's SSL cert store is misconfigured.
        try:
            proc = subprocess.run(
                ["curl", "-L", "-o", model_path, url],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0 and os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                return model_path
        except Exception:
            proc = None

        # Clean up partial downloads again.
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception:
            pass

        curl_hint = (
            f'  mkdir -p "{os.path.dirname(model_path) or "."}"\n'
            f'  curl -L -o "{model_path}" "{url}"\n'
        )
        curl_err = ""
        if proc is not None:
            curl_err = f"\n\ncurl stderr:\n{proc.stderr.strip()}\n"

        raise RuntimeError(
            "Missing MediaPipe Tasks model file and auto-download failed.\n\n"
            f"Expected model at: {model_path}\n"
            f"URL: {url}\n\n"
            "This is commonly caused by a Python SSL certificate issue (CERTIFICATE_VERIFY_FAILED).\n"
            "Workarounds:\n"
            "- Let this script download via `curl` (preferred)\n"
            "- Or download manually:\n"
            f"{curl_hint}\n"
            "- Or fix Python certificates on macOS (python.org builds): run the bundled\n"
            "  'Install Certificates.command' for your Python installation.\n"
            f"{curl_err}"
        ) from e

    return model_path


