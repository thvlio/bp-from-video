import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.fft import fft, fftfreq
import os

# Suprime avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Função para extrair o sinal PPG das bochechas
def extract_ppg_signal(frame, landmarks):
    """
    Extrai o sinal PPG a partir da região das bochechas.

    :param frame: Frame capturado pela câmera.
    :param landmarks: Pontos de referência do rosto detectados pelo Face Mesh.
    :return: Sinal PPG e coordenadas da ROI.
    """
    # Coordenadas dos landmarks das bochechas (exemplo: landmarks 234 e 454)
    cheek_left = landmarks[234]  # Ponto na bochecha esquerda
    cheek_right = landmarks[454]  # Ponto na bochecha direita

    # Define a ROI (região de interesse) ao redor das bochechas
    roi_size = 50
    x1, y1 = int(cheek_left.x * frame.shape[1]), int(cheek_left.y * frame.shape[0])
    x2, y2 = int(cheek_right.x * frame.shape[1]), int(cheek_right.y * frame.shape[0])

    roi1 = frame[y1-roi_size:y1+roi_size, x1-roi_size:x1+roi_size]
    roi2 = frame[y2-roi_size:y2+roi_size, x2-roi_size:x2+roi_size]

    # Calcula a média dos valores de cor nas ROIs (canal verde é o mais sensível para PPG)
    ppg_signal1 = np.mean(roi1[:, :, 1])  # Canal verde (índice 1)
    ppg_signal2 = np.mean(roi2[:, :, 1])  # Canal verde (índice 1)

    # Combina os sinais das duas bochechas
    ppg_signal = (ppg_signal1 + ppg_signal2) / 2

    return ppg_signal, (x1, y1, x2, y2, roi_size)


# Função para filtrar o sinal PPG
def filtrar_sinal_ppg(sinal, taxa_amostragem, lowcut=0.7, highcut=3.0):
    """
    Aplica um filtro passa-banda no sinal PPG.

    :param sinal: Lista ou array com o sinal PPG.
    :param taxa_amostragem: Taxa de amostragem do sinal (em Hz).
    :param lowcut: Frequência de corte inferior (em Hz).
    :param highcut: Frequência de corte superior (em Hz).
    :return: Sinal PPG filtrado.
    """
    # Normaliza as frequências de corte
    nyquist_freq = 0.5 * taxa_amostragem
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Projeta o filtro Butterworth passa-banda
    b, a = butter(N=4, Wn=[low, high], btype='band', analog=False)

    # Aplica o filtro
    sinal_filtrado = filtfilt(b, a, sinal)
    return sinal_filtrado


# Função para calcular a frequência cardíaca
def calcular_frequencia_cardiaca(sinal, taxa_amostragem):
    """
    Calcula a frequência cardíaca a partir do sinal PPG usando detecção de picos.

    :param sinal: Lista ou array com o sinal PPG.
    :param taxa_amostragem: Taxa de amostragem do sinal (em Hz).
    :return: Frequência cardíaca (em BPM).
    """
    # Remove a tendência do sinal
    sinal_sem_tendencia = detrend(sinal)

    # Encontra os picos no sinal PPG
    picos, _ = find_peaks(sinal_sem_tendencia, height=np.mean(sinal_sem_tendencia), distance=taxa_amostragem//2)

    # Calcula a distância média entre os picos (em segundos)
    if len(picos) >= 2:
        distancias = np.diff(picos) / taxa_amostragem
        distancia_media = np.mean(distancias)
        frequencia_cardiaca = 60 / distancia_media  # Converte para BPM
    else:
        frequencia_cardiaca = 0  # Não há picos suficientes para calcular

    return frequencia_cardiaca


# Função para estimar a pressão arterial (simulação)
def estimar_pressao_arterial(frequencia_cardiaca):
    """
    Simula a estimativa da pressão arterial com base na frequência cardíaca.
    NOTA: Esta é uma simulação e não deve ser usada para diagnóstico médico.

    :param frequencia_cardiaca: Frequência cardíaca (em BPM).
    :return: Pressão arterial sistólica e diastólica estimadas.
    """
    if frequencia_cardiaca == 0:
        return 0, 0

    # Fórmulas simuladas (apenas para fins ilustrativos)
    pressao_sistolica = 0.5 * frequencia_cardiaca + 100
    pressao_diastolica = 0.3 * frequencia_cardiaca + 60

    return pressao_sistolica, pressao_diastolica


# Listas para armazenar os dados ao longo do tempo
ppg_signals = []
bpm_list = []
pressao_sistolica_list = []
pressao_diastolica_list = []

# Inicializa o Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        exit()

    taxa_amostragem = 30  # Taxa de amostragem estimada (em Hz)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Converte o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplica equalização de histograma para melhorar o contraste
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        rgb_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Processa o frame com o Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Usa o primeiro rosto detectado
            face_landmarks = results.multi_face_landmarks[0]

            # Extrai o sinal PPG
            ppg_signal, (x1, y1, x2, y2, roi_size) = extract_ppg_signal(frame, face_landmarks.landmark)
            ppg_signals.append(ppg_signal)

            # Desenha as ROIs nas bochechas
            cv2.rectangle(frame, (x1-roi_size, y1-roi_size), (x1+roi_size, y1+roi_size), (0, 255, 0), 2)
            cv2.rectangle(frame, (x2-roi_size, y2-roi_size), (x2+roi_size, y2+roi_size), (0, 255, 0), 2)

            # Calcula a frequência cardíaca em tempo real
            if len(ppg_signals) > taxa_amostragem * 5:  # Espera 5 segundos de sinal para calcular
                ppg_filtrado = filtrar_sinal_ppg(ppg_signals, taxa_amostragem)
                frequencia_cardiaca = calcular_frequencia_cardiaca(ppg_filtrado, taxa_amostragem)
                bpm_list.append(frequencia_cardiaca)

                # Estima a pressão arterial
                pressao_sistolica, pressao_diastolica = estimar_pressao_arterial(frequencia_cardiaca)
                pressao_sistolica_list.append(pressao_sistolica)
                pressao_diastolica_list.append(pressao_diastolica)

                # Exibe a frequência cardíaca na tela
                cv2.putText(frame, f"BPM: {frequencia_cardiaca:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Exibe a pressão arterial na tela
                cv2.putText(frame, f"Pressao: {pressao_sistolica:.0f}/{pressao_diastolica:.0f} mmHg", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibe o frame
        cv2.imshow('Facial Health App', frame)

        # Salva um frame a cada 5 segundos para documentação
        if frame_count % (taxa_amostragem * 5) == 0:
            cv2.imwrite(f"frame_{frame_count}.png", frame)

        frame_count += 1

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tecla 'q' pressionada. Encerrando...")
            break

    # Libera a câmera e fecha a janela
    cap.release()
    cv2.destroyAllWindows()

# Processamento do sinal PPG e geração dos gráficos
if ppg_signals:
    # Filtra o sinal PPG
    ppg_filtrado = filtrar_sinal_ppg(ppg_signals, taxa_amostragem)

    # Gera os gráficos
    tempo = np.arange(0, len(bpm_list)) / taxa_amostragem

    # Gráfico de BPM por tempo
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, bpm_list, label="BPM", color='gray')  # Linha em cinza
    plt.title("Frequência Cardíaca (BPM) ao Longo do Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("BPM")
    plt.grid()
    plt.legend()
    # plt.savefig("bpm_vs_tempo.png")
    plt.show()

    # Gráfico de Pressão Arterial por tempo
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, pressao_sistolica_list, label="Pressão Sistólica", color='dimgray')  # Cinza escuro
    plt.plot(tempo, pressao_diastolica_list, label="Pressão Diastólica", color='lightgray')  # Cinza claro
    plt.title("Pressão Arterial ao Longo do Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Pressão (mmHg)")
    plt.grid()
    plt.legend()
    # plt.savefig("pressao_vs_tempo.png")
    plt.show()

    # Gráfico do Sinal PPG Original e Filtrado
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ppg_signals, label="Sinal PPG Original", color='gray')  # Linha em cinza
    plt.title("Sinal PPG Original")
    plt.xlabel("Tempo")
    plt.ylabel("Intensidade")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ppg_filtrado, label="Sinal PPG Filtrado", color='darkgray')  # Cinza escuro
    plt.title("Sinal PPG Filtrado")
    plt.xlabel("Tempo")
    plt.ylabel("Intensidade")
    plt.legend()

    plt.tight_layout()
    # plt.savefig("ppg_original_vs_filtrado.png")
    plt.show()

    # Gráfico de Densidade Espectral (FFT)
    n = len(ppg_filtrado)
    yf = fft(ppg_filtrado)
    xf = fftfreq(n, 1 / taxa_amostragem)

    plt.figure(figsize=(10, 5))
    plt.plot(xf[:n//2], np.abs(yf[:n//2]), label="Espectro de Frequência", color='gray')  # Linha em cinza
    plt.title("Espectro de Frequência do Sinal PPG Filtrado")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()
    # plt.savefig("espectro_frequencia.png")
    plt.show()

    # Gráfico de Distribuição de Frequência Cardíaca
    plt.figure(figsize=(10, 5))
    plt.hist(bpm_list, bins=20, color='gray', alpha=0.7)  # Barras em cinza
    plt.title("Distribuição da Frequência Cardíaca (BPM)")
    plt.xlabel("BPM")
    plt.ylabel("Frequência")
    plt.grid()
    # plt.savefig("distribuicao_bpm.png")
    plt.show()

    # Gráfico de Correlação entre BPM e Pressão Arterial
    plt.figure(figsize=(10, 5))
    plt.scatter(bpm_list, pressao_sistolica_list, label="Pressão Sistólica", color='dimgray')  # Cinza escuro
    plt.scatter(bpm_list, pressao_diastolica_list, label="Pressão Diastólica", color='lightgray')  # Cinza claro
    plt.title("Correlação entre BPM e Pressão Arterial")
    plt.xlabel("BPM")
    plt.ylabel("Pressão (mmHg)")
    plt.grid()
    plt.legend()
    # plt.savefig("correlacao_bpm_pressao.png")
    plt.show()

    # Gráfico de Tendência do Sinal PPG
    media_movel = np.convolve(ppg_filtrado, np.ones(10)/10, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(media_movel, label="Tendência do Sinal PPG", color='gray')  # Linha em cinza
    plt.title("Tendência do Sinal PPG")
    plt.xlabel("Tempo")
    plt.ylabel("Intensidade")
    plt.grid()
    plt.legend()
    # plt.savefig("tendencia_ppg.png")
    plt.show()

else:
    print("Nenhum sinal PPG foi coletado.")
