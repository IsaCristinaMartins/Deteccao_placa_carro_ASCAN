import os, re, random, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import easyocr
from pathlib import Path
from typing import Dict, List, Tuple

# ====== CONFIG ======
PASTA = Path(r"runs\detect\placas_varios_recortes\recorte_justo")
CSV_SAIDA = Path("EASY_justo_melhora_easy.csv")
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

INTERVALOS = [
    {"ini": 0, "fim": 19, "placa": "HONDUH"},
    {"ini": 21, "fim": 42, "placa": "TIMELESS"},
    {"ini": 43, "fim": 60, "placa": "EEWABUG"},
    {"ini": 61, "fim": 79, "placa": "PIKACHU"},
    {"ini": 80, "fim": 100, "placa": "OHIFIT"},
    {"ini": 101, "fim": 117, "placa": "EWGAS"},
    {"ini": 118, "fim": 145, "placa": "OFZELDA"},
    {"ini": 146, "fim": 160, "placa": "BYE4O1K"},
    {"ini": 161, "fim": 183, "placa": "MUAHAHA"},
    {"ini": 184, "fim": 202, "placa": "ORINNIE"},
    {"ini": 203, "fim": 224, "placa": "NBEYOND"},
    {"ini": 225, "fim": 243, "placa": "TOYYODA"},
    {"ini": 244, "fim": 258, "placa": "INEED2P"},
    {"ini": 259, "fim": 307, "placa": "88B88BB"},
]
# ====================

# ---------- utils ----------
def limpar(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', s.upper())

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[lb]

def acc_por_letra(pred: str, gt: str) -> float:
    pred, gt = limpar(pred), limpar(gt)
    N = max(len(pred), len(gt))
    if N == 0:
        return 0.0
    dist = levenshtein(pred, gt)
    return (1.0 - dist / N) * 100.0

def index_por_arquivo(pasta: Path) -> Dict[int, List[Path]]:
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não existe: {pasta.resolve()}")
    mapa: Dict[int, List[Path]] = {}
    for arq in pasta.glob("frame_*_*recorte_justo.*"):
        m = re.search(r"frame_(\d{6})", arq.name)
        if m:
            idx = int(m.group(1))
            mapa.setdefault(idx, []).append(arq)
    for arq in pasta.glob("frame_*recorte_justo.*"):
        m = re.search(r"frame_(\d{6})", arq.name)
        if m:
            idx = int(m.group(1))
            mapa.setdefault(idx, []).append(arq)
    return mapa

def ler_easyocr(reader: easyocr.Reader, img_bgr):
    res = reader.readtext(img_bgr, detail=1, allowlist=ALLOWLIST)
    if not res:
        up = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        res = reader.readtext(up, detail=1, allowlist=ALLOWLIST)
    if not res:
        return None, 0.0
    best = max(res, key=lambda x: x[2])
    return best[1], float(best[2])
# ---------------------------

def processar_intervalo(writer: csv.DictWriter, reader: easyocr.Reader,
                        mapa: Dict[int, List[Path]], ini: int, fim: int, placa_gt: str):
    frames_no_range = sorted([i for i in mapa.keys() if ini <= i <= fim])
    if len(frames_no_range) == 0:
        return

    escolhidos = frames_no_range if len(frames_no_range) <= 2 else random.sample(frames_no_range, 2)

    for idx in escolhidos:
        arquivo = sorted(mapa[idx])[0]
        img = cv2.imread(str(arquivo))
        if img is None:
            writer.writerow({
                "intervalo": f"{ini}-{fim}",
                "texto_bruto": "",
                "texto_limpo": "",
                "placa": placa_gt,
                "confianca_ocr": 0.0,
                "confianca_letras": 0.0,
                "confianca_placa": 0.0,
                "confianca_placa_produto": 0.0
            })
            continue

        texto_bruto, conf = ler_easyocr(reader, img)
        texto_limpo = limpar(texto_bruto) if texto_bruto else ""
        confianca_letras = acc_por_letra(texto_limpo, placa_gt)           # %
        confianca_placa = 100.0 if texto_limpo == limpar(placa_gt) and texto_limpo else 0.0

        # MÉTRICA DO ORIENTADOR: OCR [0,1] * acurácia por letra (%)
        ocr_norm = clamp(float(conf), 0.0, 1.0)                           # 0–1
        confianca_placa_produto = ocr_norm * confianca_letras             # %

        writer.writerow({
            "intervalo": f"{ini}-{fim}",
            "texto_bruto": texto_bruto or "",
            "texto_limpo": texto_limpo,
            "placa": placa_gt,
            "confianca_ocr": round(conf, 4),                              # 0–1
            "confianca_letras": round(confianca_letras, 2),               # %
            "confianca_placa": round(confianca_placa, 2),                 # % (0/100)
            "confianca_placa_produto": round(confianca_placa_produto, 2)  # % (produto)
        })

def main():
    mapa = index_por_arquivo(PASTA)
    reader = easyocr.Reader(['en'], gpu=False)

    campos = [
        "intervalo", "texto_bruto", "texto_limpo", "placa",
        "confianca_ocr", "confianca_letras", "confianca_placa",
        "confianca_placa_produto"
    ]

    with open(CSV_SAIDA, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for bloco in INTERVALOS:
            processar_intervalo(writer, reader, mapa, bloco["ini"], bloco["fim"], bloco["placa"])

    print(f"✅ CSV gerado: {CSV_SAIDA.resolve()}")

if __name__ == "__main__":
    main()
