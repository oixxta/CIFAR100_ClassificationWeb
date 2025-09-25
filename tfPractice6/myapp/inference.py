import threading
from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

# ✅ CIFAR-100 fine labels (keras 공식 순서) — y 인덱스와 1:1 대응
CLASS_NAMES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
]

_MODEL = None
_LOCK = threading.Lock()

# ⚙️ 학습과 동일한 입력 크기/전처리
IMG_SIZE = (160, 160)

def _model_path() -> Path:
    # 프로젝트 루트/tfPractice6/model/tfPractice6_CIFAR-100.keras  (경로 원하면 바꿔도 됨)
    base = Path(__file__).resolve().parent.parent   # tfPractice6/
    return base / "model" / "tfPractice6_CIFAR-100.keras"

def load_model_once():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            path = _model_path()
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            _MODEL = load_model(path)
    return _MODEL

def _preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = preprocess_input(x)            # [-1,1] 스케일 (MobileNetV2)
    x = np.expand_dims(x, axis=0)      # (1,H,W,3)
    return x

def predict_pil_image(pil_img: Image.Image, topk: int = 5):
    model = load_model_once()
    x = _preprocess_pil(pil_img)
    probs = model.predict(x, verbose=0)[0]         # (100,)
    idxs = np.argsort(probs)[::-1][:topk]
    return [{"label": CLASS_NAMES[i], "prob": float(probs[i]), "index": int(i)} for i in idxs]