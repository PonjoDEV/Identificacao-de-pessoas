import cv2
import numpy as np
import time

# Configurações iniciais
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usando DirectShow
ret, frame = cap.read()
background = None
bg_accumulated_weight = 0.5
contour_area_threshold = 500  # Área mínima para considerar como movimento
time_interval = 0.15 # Intervalo de tempo de captura em segundos

# Definir os intervalos para pessoas, crianças e animais
person_aspect_ratio_min, person_aspect_ratio_max = 1.5, 3.5
child_aspect_ratio_min, child_aspect_ratio_max = 1.0, 2.0
animal_aspect_ratio_min, animal_aspect_ratio_max = 0.5, 1.5

# Definir áreas mínimas
person_area_min, child_area_min, animal_area_min = 500, 200, 300


# Função para atualizar o background (plano de fundo estático)
def update_background(frame, bg_accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, bg_accumulated_weight)


# Função para detectar movimento
def detect_motion(frame, background):
    diff = cv2.absdiff(background.astype("uint8"), frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Contadores de vultos/pessoas
person_count = 0
child_count = 0
animal_count = 0

# Criar a janela uma vez no início
cv2.namedWindow("Detecção de Pessoas e Animais", cv2.WINDOW_NORMAL)

last_capture_time = time.time()  # Tempo da última captura

while True:
    current_time = time.time()

    # Capturar imagem a intervalo de tempo
    if current_time - last_capture_time >= time_interval:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduz a imagem para acelerar o processamento
        frame = cv2.resize(frame, (640, 480))

        # Atualizar background nos primeiros segundos
        if background is None:
            update_background(frame, bg_accumulated_weight)
        else:
            # Detectar movimento
            contours = detect_motion(frame, background)

            # Processar cada contorno detectado
            for contour in contours:
                if cv2.contourArea(contour) < contour_area_threshold:
                    continue  # Ignorar pequenos movimentos

                # Extrair o retângulo delimitador
                (x, y, w, h) = cv2.boundingRect(contour)
                aspect_ratio = h / float(w)
                area = cv2.contourArea(contour)

                # Classificar baseado no aspecto e área
                if person_area_min <= area and person_aspect_ratio_min <= aspect_ratio <= person_aspect_ratio_max:
                    person_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Pessoa', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif child_area_min <= area and child_aspect_ratio_min <= aspect_ratio <= child_aspect_ratio_max:
                    child_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Criança', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif animal_area_min <= area and animal_aspect_ratio_min <= aspect_ratio <= animal_aspect_ratio_max:
                    animal_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Animal', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Mostrar as contagens na tela
            cv2.putText(frame, f'Pessoas: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Crianças: {child_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Animais: {animal_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Atualizar o tempo da última captura
            last_capture_time = current_time

        # Mostrar o frame atual
        cv2.imshow("Detecção de Pessoas e Animais", frame)

        # Resetar contagens após cada frame
        person_count = 0
        child_count = 0
        animal_count = 0

    # Sair com a tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
