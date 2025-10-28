from pathlib import Path
import csv
import re
from paddleocr import PaddleOCR

# ======================
# CONFIGURAÇÕES
# ======================
BASE_EXEC = Path(r"C:\Users\isabel_martins\Documents\ASCAN\Comp_Cognitiva\Desafio\runs\detect\placas_varios_recortes")
PASTA = BASE_EXEC / "recorte_justo"
SAIDA_CSV = BASE_EXEC / "resultado_ocr_paddle_justo.csv"

# OCR (compatível com 2.x e 3.x)
# use_angle_cls melhora quando a placa está ligeiramente inclinada
ocr = PaddleOCR(lang='en', use_angle_cls=True)  # não passe det/rec/show_log

# Ground-truth por intervalo de frame
PLACAS_GT = [
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
    {"ini": 259, "fim": 307, "placa": "88B88BB"}
]

def obter_placa_real(frame_num: int) -> str:
    for it in PLACAS_GT:
        if it["ini"] <= frame_num <= it["fim"]:
            return it["placa"]
    return ""

def extrair_numero_frame(nome: str) -> int:
    m = re.search(r"frame_(\d+)", nome)
    return int(m.group(1)) if m else -1

def limpar_texto(txt: str) -> str:
    import re as _re
    txt = (txt or "").strip().upper()
    return _re.sub(r"[^A-Z0-9]", "", txt)

def acuracia_por_caractere(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    iguais = sum(p == g for p, g in zip(pred, gt))
    return iguais / len(gt)

def acuracia_por_placa(pred: str, gt: str) -> int:
    return 1 if pred == gt else 0

# --------- Parser robusto (2.x e 3.x) ----------
def parse_ocr_result(res):
    """
    Suporta:
      - 2.x: [[ [box], (text, score) ], ...]
      - 3.x: pode retornar list[dict] ou dict com rec_texts/rec_scores
    Retorna (texto_concatenado, conf_media)
    """
    textos, confs = [], []

    if res is None or res == []:
        return "", -1.0

    # Se vier no formato "lista com uma sublista", use o primeiro nível útil
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        res = res[0]

    if isinstance(res, dict):
        # 3.x: dict com chaves de reconhecimento direto
        if "rec_texts" in res:
            textos = [t for t in res.get("rec_texts", []) if isinstance(t, str)]
            confs = [float(s) for s in res.get("rec_scores", []) if isinstance(s, (int, float))]
        else:
            t = res.get("text") or res.get("transcription")
            s = res.get("score") or res.get("confidence") or res.get("probability")
            if isinstance(t, str) and t:
                textos.append(t)
            if isinstance(s, (int, float)):
                confs.append(float(s))
    elif isinstance(res, list):
        for item in res:
            # 2.x: [box, (text, score)]
            if (isinstance(item, (list, tuple)) and len(item) >= 2
                and isinstance(item[1], (list, tuple)) and len(item[1]) == 2):
                t, s = item[1]
                if isinstance(t, str) and t:
                    textos.append(t)
                try:
                    confs.append(float(s))
                except:  # noqa
                    pass
            # 3.x: dicts por item
            elif isinstance(item, dict):
                t = item.get("text") or item.get("transcription")
                s = item.get("score") or item.get("confidence") or item.get("probability")
                if isinstance(t, str) and t:
                    textos.append(t)
                if isinstance(s, (int, float)):
                    confs.append(float(s))

    texto = "".join(textos) if textos else ""
    conf = (sum(confs) / len(confs)) if confs else -1.0
    return texto, conf

# --------- OCR (chamada compatível 2.x/3.x) ----------
def run_ocr_on_path(img_path: str):
    """
    Tenta primeiro com 'cls=True' (padrão do exemplo da sua amiga / 2.x).
    Se a versão não aceitar, faz fallback sem 'cls'.
    """
    try:
        res = ocr.ocr(img_path, cls=True)
    except TypeError:
        res = ocr.ocr(img_path)
    return parse_ocr_result(res)

# ======================
# EXECUÇÃO
# ======================
if not PASTA.exists():
    raise FileNotFoundError(f"Pasta não encontrada: {PASTA}")

# Extensões comuns
exts = ("*.jpg", "*.jpeg", "*.png")
arquivos = []
for e in exts:
    arquivos.extend(PASTA.glob(e))
arquivos = sorted(arquivos)
if not arquivos:
    raise FileNotFoundError(f"Nenhuma imagem encontrada em {PASTA} com {exts}")

with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["arquivo", "texto_bruto", "texto_limpo", "conf_media",
                "placa_real", "acuracia_caractere", "acuracia_placa"])

    for p in arquivos:
        # Use o caminho direto (evita img=None do OpenCV)
        texto_bruto, conf = run_ocr_on_path(str(p))
        texto_limpo = limpar_texto(texto_bruto)

        frame = extrair_numero_frame(p.name)
        gt = obter_placa_real(frame)

        ac_char = acuracia_por_caractere(texto_limpo, gt)
        ac_full = acuracia_por_placa(texto_limpo, gt)

        w.writerow([p.name, texto_bruto or "", texto_limpo, f"{conf:.3f}",
                    gt, f"{ac_char:.3f}", ac_full])

print(f"\n✅ OCR concluído! CSV salvo em: {SAIDA_CSV}")
