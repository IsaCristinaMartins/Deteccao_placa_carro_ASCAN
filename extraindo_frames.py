import os
import cv2



# Função de extração de frames de vídeo

def executar_extracao():
    caminho_video = "data/videos/video_placa_OK.mp4" 
    pasta_saida = "imagens_frame/quadro"
    os.makedirs(pasta_saida, exist_ok=True)

    captura = cv2.VideoCapture(caminho_video) # Captura o vídeo 
    fps = captura.get(cv2.CAP_PROP_FPS) or 30  # Começa os cortes
    if fps == 0:
        fps = 30

    quadros_por_segundo = 10
    intervalo = int(fps / quadros_por_segundo)

    numero_quadro = 0
    salvos = 0

    while captura.isOpened():  # Loop 
        sucesso, quadro = captura.read()
        if not sucesso:
            break
        if numero_quadro % intervalo == 0:
            nome_arquivo = os.path.join(pasta_saida, f"frame_{numero_quadro}.jpg")
            cv2.imwrite(nome_arquivo, quadro)
            salvos += 1
        numero_quadro += 1

    captura.release()
    print(f"[INFO] Quadros salvos: {salvos}")

if __name__ == "__main__":
    executar_extracao()


# Ta dando um erro, porque sai que foram salvos 639 quadros e na pasta tem muito mais. 