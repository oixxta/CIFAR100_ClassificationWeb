import os
from django.apps import AppConfig

class MyappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "myapp"

    def ready(self):
        # runserver의 auto-reload로 두 번 로딩되는 것 방지
        if os.environ.get("RUN_MAIN") == "true":
            from .inference import load_model_once
            try:
                load_model_once()
                print("[myapp] Model loaded.")
            except Exception as e:
                print(f"[myapp] Model load deferred or failed: {e}")