# grafico_similaridade.py
# Gera CSV e gráficos (SSIM/MSE/Laplaciano) para os crops de placa

from pathlib import Path
import numpy as np
import cv2, csv, os
import matplotlib.pyplot as plt

# SSIM (skimage)
try:
    from skimage.metrics import structural_similarity as ssim
    SKI = True
except Exception:
    SKI = False

# === AJUSTE ESTE CAMINHO PARA A SUA PASTA DE CROPS ===
PASTA_CROPS = Path(r"C:\Users\isabel_martins\Documents\ASCAN\Computação Cognitiva\Desafio\runs\detect\predict_python\crops\placa")
ARQ_METRICAS = PASTA_CROPS / "metricas_similaridade.csv"
FIG_MUDANCAS = PASTA_CROPS / "grafico_mudancas.png"
FIG_NITIDEZ  = PASTA_CROPS / "grafico_nitidez.png"

# ---------- utils ----------
def imread_unicode(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def to_gray01(img, size=(256,128)):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)/255.0

def to_rgb01(img, size=(256,128)):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32)/255.0

def var_laplacian(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def ssim_gray(a, b):
    A, B = to_gray01(a), to_gray01(b)
    if SKI:
        return float(ssim(A, B, data_range=1.0))
    # fallback simples: aproxima com 1 - MSE
    return max(0.0, 1.0 - float(np.mean((A - B)**2)))

def ssim_color(a, b):
    if SKI:
        A, B = to_rgb01(a), to_rgb01(b)
        try:
            return float(ssim(A, B, data_range=1.0, channel_axis=-1))
        except TypeError:  # skimage antigo
            return float(ssim(A, B, data_range=1.0, multichannel=True))
    # fallback: SSIM no canal V (HSV)
    A = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[:,:,2]
    B = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:,:,2]
    A = cv2.resize(A, (256,128), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    B = cv2.resize(B, (256,128), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    if SKI:
        return float(ssim(A, B, data_range=1.0))
    return max(0.0, 1.0 - float(np.mean((A - B)**2)))

# ---------- main ----------
def main():
    if not PASTA_CROPS.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {PASTA_CROPS}")

    imgs_paths = sorted([p for p in PASTA_CROPS.iterdir()
                         if p.suffix.lower() in (".jpg",".jpeg",".png")], key=lambda p: p.name)
    if not imgs_paths:
        print("Nenhum crop encontrado.")
        return

    # Referência: o frame mais nítido (melhor para comparar)
    nitidez_list = []
    imgs_cache = []
    for p in imgs_paths:
        im = imread_unicode(p)
        imgs_cache.append(im)
        nitidez_list.append(var_laplacian(im))
    ref_idx = int(np.argmax(nitidez_list))
    ref_img = imgs_cache[ref_idx]

    # Métricas vs referência e vs anterior (delta temporal)
    rows = []
    prev_img = None
    for i, (p, im) in enumerate(zip(imgs_paths, imgs_cache)):
        gA, gB = to_gray01(ref_img), to_gray01(im)
        mse_gray_ref = float(np.mean((gA - gB)**2))
        ssim_g_ref   = ssim_gray(ref_img, im)
        ssim_c_ref   = ssim_color(ref_img, im)
        sharp        = var_laplacian(im)

        # também calcular contra anterior (se existir)
        if prev_img is not None:
            gP, gB2 = to_gray01(prev_img), to_gray01(im)
            mse_gray_prev = float(np.mean((gP - gB2)**2))
            ssim_g_prev   = ssim_gray(prev_img, im)
        else:
            mse_gray_prev = np.nan
            ssim_g_prev   = np.nan

        rows.append({
            "idx": i,
            "arquivo": p.name,
            "ref_idx": ref_idx,
            "mse_gray_ref": mse_gray_ref,
            "ssim_gray_ref": ssim_g_ref,
            "ssim_color_ref": ssim_c_ref,
            "mse_gray_prev": mse_gray_prev,
            "ssim_gray_prev": ssim_g_prev,
            "lap_var": sharp
        })
        prev_img = im

    # Salva CSV
    ARQ_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQ_METRICAS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --------- Gráficos ----------
    idx = [r["idx"] for r in rows]
    inv_ssim_gray = [1.0 - r["ssim_gray_ref"] for r in rows]
    inv_ssim_color= [1.0 - r["ssim_color_ref"] for r in rows]
    mse_gray      = [r["mse_gray_ref"] for r in rows]

    plt.figure(figsize=(12,6))
    plt.plot(idx, inv_ssim_gray, label="1 - SSIM (cinza) vs ref")
    plt.plot(idx, inv_ssim_color, label="1 - SSIM (cor) vs ref")
    plt.plot(idx, mse_gray, label="MSE (cinza) vs ref")
    plt.xlabel("Índice do frame (crop ordenado)")
    plt.ylabel("Distância / diferença")
    plt.title("Mudanças entre frames (picos = mudança real)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_MUDANCAS, dpi=160)

    # Nitidez
    lap = [r["lap_var"] for r in rows]
    plt.figure(figsize=(12,4))
    plt.plot(idx, lap, label="Variância do Laplaciano (nitidez)")
    plt.xlabel("Índice do frame (crop ordenado)")
    plt.ylabel("Nitidez (maior é melhor)")
    plt.title("Nitidez por frame (escolha p/ OCR nos picos de nitidez)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_NITIDEZ, dpi=160)

    print("✅ Métricas salvas em:", ARQ_METRICAS)
    print("🖼️ Gráficos salvos em:")
    print("   -", FIG_MUDANCAS)
    print("   -", FIG_NITIDEZ)
    if not SKI:
        print("⚠️ Observação: scikit-image não encontrado. Instale para SSIM em cor real:  pip install scikit-image")

if __name__ == "__main__":
    main()
