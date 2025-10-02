import os
import glob
import cv2


# Função de extração de frames de vídeo
def executar_extracao():
    # Path
    caminho_video = "data/videos/video_placa_OK.mp4" 
    pasta_saida = "imagens_frame/quadro"
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Captura em si
    captura = cv2.VideoCapture(caminho_video)
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30  
    # Garantindo o intervalo que vão cortar
    quadros_por_segundo = 10
    intervalo = max(1, round(fps / quadros_por_segundo))  

    numero_quadro = 0
    salvos = 0  
    # Loop para nomear e salvar o frame
    while captura.isOpened():
        sucesso, quadro = captura.read()
        if not sucesso:
            break

        if numero_quadro % intervalo == 0:
            nome_arquivo = os.path.join(pasta_saida, f"frame_{salvos:06d}.jpg")
            ok = cv2.imwrite(nome_arquivo, quadro)
            if ok:
                salvos += 1

        numero_quadro += 1

    captura.release()

    # Contando pra não da erro (Jesus eu tô aqui, viu?)
    contagem_no_disco = len(glob.glob(os.path.join(pasta_saida, "frame_*.jpg")))
    print(f"[INFO] Quadros salvos: {contagem_no_disco}")
    return contagem_no_disco

if __name__ == "__main__":
    executar_extracao()
