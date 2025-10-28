# melhora_easyocr_recorte_justo.py
# Basta rodar:  python melhora_easyocr_recorte_justo.py
# Lê imagens em runs\detect\placas_varios_recortes\recorte_justo
# Salva melhoradas em runs\detect\placas_varios_recortes\recorte_justo_easyocr

import cv2
import numpy as np
from pathlib import Path

# === PASTAS FIXAS (altere se precisar) ===
BASE_INP  = Path("runs") / "detect" / "placas_varios_recortes" / "recorte_justo"
BASE_OUT  = Path("runs") / "detect" / "placas_varios_recortes" / "recorte_justo_easyocr"
TARGET_H  = 960  # altura alvo p/ upsample leve

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resize_keep_ar(img, target_h=960):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_h / float(h)
    if scale <= 1.0:
        scale = max(scale, 1.2)  # pequeno upsample ainda assim
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def gray_world_white_balance(img):
    img = img.astype(np.float32)
    B, G, R = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    avg = (B + G + R) / 3.0 + 1e-6
    img[:,:,0] *= (avg / (B + 1e-6))
    img[:,:,1] *= (avg / (G + 1e-6))
    img[:,:,2] *= (avg / (R + 1e-6))
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_on_l(img_bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def auto_gamma(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(np.float32) / 255.0
    mean_v = np.clip(v.mean(), 1e-3, 0.999)
    gamma = 0.7 if mean_v < 0.45 else (1.2 if mean_v > 0.75 else 1.0)
    if abs(gamma - 1.0) < 1e-3:
        return img_bgr
    lut = np.array([(i/255.0)**gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, lut)

def denoise(img_bgr):
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)

def unsharp(img_bgr, strength=0.5, radius=1.8):
    blur = cv2.GaussianBlur(img_bgr, (0,0), radius)
    sharp = cv2.addWeighted(img_bgr, 1+strength, blur, -strength, 0)
    return sharp

def enhance_for_easyocr(img_bgr, target_h=960):
    wb = gray_world_white_balance(img_bgr)
    dn = denoise(wb)
    cl = clahe_on_l(dn, clip=2.0, tiles=8)
    gm = auto_gamma(cl)
    sh = unsharp(gm, strength=0.5, radius=1.8)
    rs = resize_keep_ar(sh, target_h=target_h)
    return rs

def main():
    ensure_dir(BASE_OUT)
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

    imgs = [p for p in BASE_INP.rglob("*") if p.suffix.lower() in exts]
    if not imgs:
        print(f"[AVISO] Nenhuma imagem em: {BASE_INP.resolve()}")
        return

    ok, fail = 0, 0
    for p in imgs:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[ERRO] não abriu: {p}")
                fail += 1
                continue
            enh = enhance_for_easyocr(img, target_h=TARGET_H)
            rel = p.relative_to(BASE_INP)
            dst = (BASE_OUT / rel).with_suffix(".png")
            ensure_dir(dst.parent)
            cv2.imwrite(str(dst), enh)
            print(f"[OK] {rel} -> {dst.relative_to(BASE_OUT)}")
            ok += 1
        except Exception as e:
            print(f"[FALHA] {p}: {e}")
            fail += 1

    print(f"\nConcluído. Sucesso: {ok} | Falhas: {fail}")
    print(f"Saída em: {BASE_OUT.resolve()}")

if __name__ == "__main__":
    main()
