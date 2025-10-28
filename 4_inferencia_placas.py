from ultralytics import YOLO
from pathlib import Path
import os
import csv
import cv2
import numpy as np
import torch

# ==========================
# CONFIGURAÇÕES DO USUÁRIO
# ==========================
pasta_frames = r"imagens_frame\\quadro"
modelo_peso  = r"runs\\carplates2\\weights\\best.pt"
diretorio_projeto = r"C:\\Users\\isabel_martins\\Documents\\ASCAN\\Comp_Cognitiva\\Desafio\\runs\\detect"
nome_execucao = "placas_varios_recortes"

conf_minima = 0.25
imgsz = 960
device = (0 if torch.cuda.is_available() else "cpu")
usar_half = bool(torch.cuda.is_available())
max_det = 5

# ==========================
# FUNÇÕES AUXILIARES
# ==========================
def clamp(val, low, high):
    return max(low, min(high, val))

def expand_laterais(x1, y1, x2, y2, pad, W, H):
    nx1 = clamp(int(x1) - pad, 0, W - 1)
    nx2 = clamp(int(x2) + pad, 0, W - 1)
    ny1 = clamp(int(y1), 0, H - 1)
    ny2 = clamp(int(y2), 0, H - 1)
    if ny2 <= ny1: ny2 = min(H - 1, ny1 + 1)
    if nx2 <= nx1: nx2 = min(W - 1, nx1 + 1)
    return nx1, ny1, nx2, ny2

def preprocess_otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() < 127:
        b = cv2.bitwise_not(b)
    return b

def preprocess_adapt(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    return b

# ==========================
# PREPARO DE SAÍDAS
# ==========================
exec_dir = Path(diretorio_projeto) / nome_execucao
exec_dir.mkdir(parents=True, exist_ok=True)

recortes = {
    0: "recorte_justo",
    8: "recorte_Oito",
    16: "recorte_dezesseis",
    32: "recorte_TrintaDois"
}

for pad, nome in recortes.items():
    (exec_dir / nome).mkdir(exist_ok=True)
    (exec_dir / f"{nome}LimpoOtsu").mkdir(exist_ok=True)
    (exec_dir / f"{nome}LimpoAdapt").mkdir(exist_ok=True)

# ==========================
# INFERÊNCIA YOLO
# ==========================
modelo = YOLO(modelo_peso)
resultados = modelo.predict(
    source=pasta_frames,
    conf=conf_minima,
    imgsz=imgsz,
    device=device,
    half=usar_half,
    max_det=max_det,
    save=False,
    project=str(exec_dir),
    name="_tmp_ignore",
    exist_ok=True,
    verbose=False,
)

names = resultados[-1].names if resultados else {}

def descobrir_id_classe_placa(names: dict) -> int | None:
    alvo = {"plate","placa","license-plate","licence-plate","license_plate","car-plate","car_plate"}
    for k, v in names.items():
        if str(v).strip().lower().replace(" ", "-").replace("_", "-") in alvo:
            return int(k)
    if len(names) == 1:
        return int(list(names.keys())[0])
    return None

id_placa = descobrir_id_classe_placa(names)

# ==========================
# PROCESSAR DETECÇÕES
# ==========================
for pad, nome in recortes.items():
    csv_path = exec_dir / f"{nome}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame","det_idx","classe_id","classe_nome","conf","x1","y1","x2","y2","arquivo_bruto","arquivo_otsu","arquivo_adapt"])

        for r in resultados:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            frame_path = Path(r.path)
            frame_name = frame_path.name
            img = r.orig_img
            H, W = img.shape[:2]

            cls_list = r.boxes.cls.tolist()
            conf_list = r.boxes.conf.tolist()
            xyxy_list = r.boxes.xyxy.tolist()

            det_idx = 0
            for c, conf, (x1, y1, x2, y2) in zip(cls_list, conf_list, xyxy_list):
                c_int = int(c)
                if id_placa is not None and c_int != id_placa:
                    continue

                ex1, ey1, ex2, ey2 = expand_laterais(x1, y1, x2, y2, pad, W, H)
                crop = img[int(ey1):int(ey2), int(ex1):int(ex2)].copy()

                base_out = f"{frame_path.stem}_d{det_idx}_{nome}.jpg"

                # salvar recorte bruto
                out_dir = exec_dir / nome
                cv2.imwrite(str(out_dir / base_out), crop)

                # preprocessar e salvar Otsu
                limpo_otsu = preprocess_otsu(crop)
                cv2.imwrite(str(exec_dir / f"{nome}LimpoOtsu" / base_out), limpo_otsu)

                # preprocessar e salvar Adaptativo
                limpo_adapt = preprocess_adapt(crop)
                cv2.imwrite(str(exec_dir / f"{nome}LimpoAdapt" / base_out), limpo_adapt)

                writer.writerow([
                    frame_name,
                    det_idx,
                    c_int,
                    names.get(c_int, str(c_int)),
                    float(conf),
                    int(x1), int(y1), int(x2), int(y2),
                    base_out,
                    base_out,
                    base_out
                ])

                det_idx += 1

print("\n✅ Detecção concluída, recortes salvos com versões Otsu e Adapt. Um CSV separado foi criado para cada variante de recorte.")
