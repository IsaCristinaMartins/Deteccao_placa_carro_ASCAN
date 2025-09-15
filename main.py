import os
import cv2


# Variáveis iniciais para caminho do vídeo e alocação de frames
caminho_video = "data/videos/qqvideoseila.mp4"
pasta_saida = "imagens_frame/quadro"
os.makedirs(pasta_saida, exist_ok=True) #Criação de pasta caso ela não exista




# Criação de variável para captura de frame
captura = cv2.VideoCapture(caminho_video)
fps = captura.get(cv2.CAP_PROP_FPS) # Lê a taxa real de quadros

if fps ==0:
    fps = 30 #Valor de segurança

quadros_por_segundo = 10 # Quantidade que quero salvar por segundo
intervalo = int(fps / quadros_por_segundo) # A cada quantos quadros salvar

numero_quadro = 0
salvos = 0


# Loop para abertura de vídeo e captura de frame com tratamento de erro caso não tenha sucesso.
while captura.isOpened():
    sucesso, quadro = captura.read()
    if not sucesso:
        break
    
    if numero_quadro % intervalo == 0:
        nome_arquivo = os.path.join(pasta_saida, f"frame_{numero_quadro}.jpg")
        cv2.imwrite(nome_arquivo, quadro)
        salvos +=1

    
    numero_quadro +=1

captura.release()

print(f"[INFO] Quadros salvos: {salvos}")
