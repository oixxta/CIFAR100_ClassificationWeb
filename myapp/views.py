import io, time
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt   # 데모용(실서비스는 토큰 권장)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .inference import predict_pil_image

def index(request):
    return render(request, "myapp/index.html")

@csrf_exempt   # 데모 편의용. 실제론 CSRF 토큰 세팅 권장.
@require_POST
def predict(request):
    f = request.FILES.get("image")
    if not f:
        return JsonResponse({"error": "no_file"}, status=400)

    # 저장 경로
    media_dir = Path(settings.MEDIA_ROOT) / "uploads"
    media_dir.mkdir(parents=True, exist_ok=True)
    img_path = media_dir / f"{int(time.time()*1000)}_{f.name}"
    with open(img_path, "wb") as out:
        for chunk in f.chunks():
            out.write(chunk)

    # PIL 열기
    pil_img = Image.open(img_path)

    # (내부 확인용) matplotlib 시각화 저장
    debug_dir = Path(settings.MEDIA_ROOT) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    dbg_path = debug_dir / (img_path.stem + "_debug.png")
    plt.figure()
    plt.imshow(pil_img.convert("RGB"))
    plt.axis("off")
    plt.title("Uploaded Image (server-side debug)")
    plt.savefig(dbg_path, bbox_inches="tight")
    plt.close()

    # 추론
    topk = predict_pil_image(pil_img, topk=5)
    top1 = topk[0]["label"]

    # 클라이언트엔 결과(문자열/Top-5)만 반환
    return JsonResponse({
        "result": top1,
        "top5": topk,
        # "debug_image_url": request.build_absolute_uri(settings.MEDIA_URL + f"debug/{dbg_path.name}")  # 필요시 확인용 노출
    })