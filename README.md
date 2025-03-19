# bp-from-video

Projeto final de Transdução e Instrumentação Biomédica (PPGEB - UnB). O projeto consiste em estimar a pressão sanguínea de uma pessoa com base em vídeo do rosto (gravado ou ao vivo).

## Abordagem

O algoritmo consiste num loop de leitura de frames, processamento de frames, processamento de sinais e exibição dos resultados na tela.

A primeira parte do algoritmo consiste em abertura de vídeo ou feed com OpenCV. O leitor passa por uma breve calibração para permitir que a câmera se auto ajuste, mas desativa os ajustes automáticos depois de alguns segundos. Depois de lido um frame, são realizadas inferências com até 4 modelos implementados:
- FaceDetector;
- FaceLandmarker;
- HandLandmarker;
- PersonSegmenter.

São todos do MediaPipe, uma vez que um dos objetivos iniciais do trabalho era que o algoritmo pudesse ser portado para o Android. É realizado um breve aquecimento da inferência para que se elimine o atraso inicial da inferência do período de amostragem do sinal. As detecções são exibidas na tela, incluindo os landmarks individuais. Depois disso, são lidas as configurações de regiões de interesse, que ditam que landmarks específicos e de quais modelos serão usados para cada região e como será o formato da região. Atualmente, estão configuradas as seguintes regiões:
- Logo acima da sobrancelha;
- No centro da palma da mão.

Depois de obtidos os bounding boxes das regiões, os bounding boxes são opcionalmente filtrados no tempo e em seguida servem para a amostragem dos sinais de iPPG em cada região de interesse. As regiões de interesse são exibidas na tela. É possível escolher entre formas diferentes de se amostrar o iPPG na região de interesse:
- Média do canal verde;
- Média da crominância verde.

O valor amostrado é adicionado a um buffer limitado do sinal iPPG no tempo. Em seguida, algumas técnicas de processamento de sinais podem ser combinadas e aplicadas, tais como:
- Derivada dicreta de 1º e 2º grau;
- Interpolação linear e por spline;
- Subtração de tendência;
- Filtros Butterworth e FIR.

A interpolação se faz necessária uma vez que muitos métodos pressupõem sinais com período de amostragem constante. As frequências de corte dos filtros são previamente definidas como sendo as frequências cardíacas mínima e máxima do coração humano. O sinal é construído, mas não é exibido na tela em seu estado cru.

Assim, o sinal processado é transformado para o domínio da frequência. Estão disponíveis as seguintes transformações:
- Transformada discreta de Fourier;
- Periodograma de Welch;
- Periodograma de Lomb-Scargle.

Então, é possível encontrar a frequência que possui maior energia no sinal. Com esta frequência é estimada a frequência cardíaca. Um filtro temporal pode ser aplicado à frequência cardíaca, e seu valor é exibido na tela.

Por fim, é realizada uma estimativa do tempo de trânsito do pulso (PTT) entre combinações dos sinais das regiões configuradas. Uma correlação é feita entre os sinais e o tempo de atraso com maior correlação é obtido, sendo este o PTT. Com o PTT, tínhamos o objetivo de estimar a pressão sanguínea com algum método de regressão, mas o objetivo foi paralisado.

A exibição dos gráficos em tempo real foi feita com o OpenCV, com funções próprias desenvolvidas para este trabalho. Matplotlib foi evitado já que seu desempenho para exibição de gráficos em tempo real é bastante limitada.

## Resultados e Limitações

A estimativa da frequência cardíaca funciona bem, principalmente com o periodograma de Lomb-Scargle, sem necessariamente usar interpolação para uniformização do período de amostragem. Os filtros se mostram suficientes para eliminar as frequências irrelevantes, e o pico costuma ficar visível depois que a pessoa fica parada em frente à câmera por alguns segundos. Quanto mais iluminação natural, melhor. Com um computador suficientemente capaz, foi possível manter a frequência de amostragem real perto de 25 Hz (comparado à frequência máxima de 30 Hz da câmera utilizada). Assim, não há quase nenhum risco de aliasing, mesmo que se escolhesse uma frequência cardíaca máxima de 4 Hz. Se for possível manter a frequência de amostragem acima de 8 Hz, não haverá aliasing. Ainda há espaço para determinar qual é a melhor forma de processar o sinal cru.

A estimativa da pressão sanguínea, por outro lado, não seria viável com a câmera utilizada (de 30 FPS), e ainda seria restrita em vídeos gravados com 60 FPS. Mais tarde no projeto, percebi que a maioria dos estudos que utiliza o PTT usa câmeras de alta velocidade, como câmeras de 120 FPS. Isto é necessário porque alguns estudos similares, que usaram esta abordagem e também usaram o rosto e a mão como região de interesse, observaram PTTs (entre rosto e mão) que variam entre 10 ms e 60 ms. Se considerarmos que o pulso do sangue é um sinal por si só, temos frequências de corte de 17 Hz e 100 Hz. Para evitar o risco de aliasing, é necessário frequência de amostragem que seja o dobro da frequência máxima da banda do sinal, então teríamos que ter frequência de amostragem de 200 Hz (uma câmera de 200 FPS). Com 30 FPS, a câmera utilizada suportaria, sem aliasing, banda de frequência indo até no máximo 15 Hz, ou no mínimo 67 ms. Além disso, um filtro passa banda teria que ser utilizado além ou ao invés do filtro do estágio anterior, que corta frequências acima de 4 Hz. Sendo assim, o método não funciona corretamente, e os valores de PTT flutuam bastante.

Existem abordagens surgindo que utilizas machine learning para estimar a pressão sanguínea com somente uma região de interesse. Esta abordagem seria mais adequada aos equipamentos disponíveis, mas necessitaria investigar bancos de dados prontos ou montar um banco que tenha o iPPG e dados confiáveis de pressão sanguínea, ambos sincronizados no tempo. Pode-se também usar bancos que tenham PPG e pressão e aplicar técnicas de transfer learning. Não havia tempo hábil para este tipo de abordagem no decorrer do semestre.

## Instalação

Recomendo que seja testado em Python >= 3.12. Recomendo criação de ambiente virtual e instalação dos requisitos em requirements.txt.

Os algoritmos foram testados no Ubuntu 24.04 com uma GPU dedicada (RTX 3060 Ti). A API de vídeo disponível ao OpenCV pode não ser a mesma em outros sistemas operacionais.

## Execução

Primeiro, ajuste os parâmetros dentro dos scripts, se necessário. Depois, execute:
```
python bp.py
```

Para a versão que usa multiprocessing, execute:
```
python pbp.py
```
