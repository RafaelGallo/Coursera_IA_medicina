# Coursera IA Medicina

[![author](https://img.shields.io/badge/author-Rafael_Gallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html) 
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-green.svg?style=flat)]([https://github.com/danielesantiago/Data-Science](https://github.com/RafaelGallo?tab=repositories))
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)


# Curso 1 - IA para diagnóstico médico
## Projeto semana 1 - Diagnóstico médico de radiografia de tórax com aprendizado profundo

**Descrição projeto**

Este projeto é uma compilação de vários subprojetos do curso IA para Especialização Médica da Coursera, que tem como objetivo usar um modelo de aprendizado profundo para diagnosticar patologias em radiografias de tórax. O projeto utiliza um modelo DenseNet-121 pré-treinado capaz de diagnosticar 14 rótulos, como Cardiomegalia, Massa, Pneumotórax ou Edema. Em outras palavras, esse único modelo pode fornecer previsões de classificação binária para cada uma das 14 patologias rotuladas. 
A normalização de peso é realizada para compensar a baixa prevalência das anormalidades nos conjuntos de dados de radiografias de tórax (desequilíbrio de classes). Por fim, a técnica GradCAM é utilizada para destacar e visualizar onde o modelo está focando, ou seja, qual área de interesse é usada para fazer a previsão. Essa é uma ferramenta que pode ser útil para a descoberta de marcadores, análise de erros, treinamento e até mesmo na implantação do modelo.

## Base dados

O projeto utiliza imagens de radiografias de tórax retiradas do conjunto de dados público ChestX-ray8. Este conjunto de dados contém 108.948 imagens de radiografias de tórax em vista frontal de 32.717 pacientes únicos. Cada imagem no conjunto de dados contém várias etiquetas identificando 14 diferentes condições patológicas, extraídas por mineração de texto. Essas etiquetas podem, por sua vez, ser usadas por médicos para diagnosticar 8 doenças diferentes. Para o projeto, trabalhamos com um subconjunto de aproximadamente 1000 imagens.

* 875 imagens são usadas para treinamento.
* 109 imagens são usadas para validação.
* 420 imagens são usadas para teste.

O conjunto de dados inclui um arquivo CSV que fornece as etiquetas verdadeiras para cada radiografia.

# Modelo rede neural

## DenseNet highlights

O DenseNet, abreviação para Redes Convolucionais Densamente Conectadas, surgiu em 2017 por meio de um artigo premiado de autoria de Gao Huang e sua equipe, datado de 2018. Este inovador modelo demonstrou a capacidade de superar arquiteturas anteriores, incluindo o ResNet, que eu expliquei detalhadamente em um projeto anterior focado na dermatologia de IA para o diagnóstico de câncer de pele. Independentemente das nuances nas estruturas arquitetônicas dessas redes, todas compartilham um objetivo comum: estabelecer canais para a livre circulação de informações entre as camadas iniciais e finais. 

![img1](https://github.com/RafaelGallo/Coursera_IA_medicina/assets/44452165/09a36fcc-68c2-4157-8aaf-d5bd3fead1eb)

O DenseNet, seguindo essa mesma diretriz, estabelece conexões diretas entre as camadas da rede, criando, assim, uma rede densamente conectada. Partes deste resumo estão disponíveis em detalhes em uma revisão mais ampla que pode ser consultada para obter informações adicionais. A principal novidade do DenseNet: o DenseNet é uma rede convolucional na qual cada camada está conectada a todas as outras camadas que estão mais profundas na rede. A primeira camada está conectada à segunda, terceira, quarta, e assim por diante. A segunda camada está conectada à terceira, quarta, quinta, e assim por diante. Cada camada em um bloco denso recebe mapas de características de todas as camadas anteriores e envia sua saída para todas as camadas subsequentes. Os mapas de características recebidos de outras camadas são fundidos por meio de concatenação, e não por soma (como nas ResNets). Os mapas de características extraídos são continuamente adicionados aos anteriores, o que evita trabalho redundante e duplicado.

Isso permite que a rede reutilize informações já aprendidas e seja mais eficiente. Essas redes requerem menos camadas. Resultados de ponta são alcançados com apenas 12 mapas de características de canal. Isso também significa que a rede tem menos parâmetros para aprender e, portanto, é mais fácil de treinar. Entre todas as variantes, o DenseNet-121 é o padrão.

**Principais contribuições da arquitetura DenseNet**

* Alivia o problema do gradiente desvanecido (à medida que as redes ficam mais profundas, os gradientes não são retropropagados de forma suficiente para as camadas iniciais da rede. Os gradientes continuam diminuindo à medida que retrocedem na rede e, como resultado, as camadas iniciais perdem sua capacidade de aprender as características básicas de baixo nível).

* Propagação mais forte de características.
 
A camada CONV corresponde à sequência BatchNorm->ReLU->Conv. Cada camada tem essa sequência repetida duas vezes, a primeira com convolução 1x1, produzindo: taxa de crescimento x 4 mapas de características, a segunda com convolução 3x3. Os autores descobriram que o modo de pré-ativação (BN e ReLU antes da Conv) era mais eficiente do que o modo de pós-ativação usual. A taxa de crescimento (k = 32 para o DenseNet-121) define o número de mapas de características de saída de uma camada. Basicamente, as camadas geram 32 mapas de características que são adicionados a um número de 32 mapas de características das camadas anteriores. Enquanto a profundidade aumenta continuamente, cada camada traz a profundidade de volta para 32.

* Camada de Transição: Entre os blocos densos, você encontra a camada de transição. Em vez de somar o residual como na ResNet, o DenseNet concatena todos os mapas de características. Uma camada de transição é composta por: Normalização em Lote (Batch Normalization) -> Convolução 1x1 -> Pooling Médio (Average Pooling). As camadas de transição entre dois blocos densos garantem o papel de down-sampling (diminuição das dimensões x e y pela metade), essencial para redes neurais convolucionais (CNN). As camadas de transição também comprimem o mapa de características e reduzem os canais pela metade. Isso contribui para a compacidade da rede.

# Arquitetura DenseNet

A arquitetura DenseNet é composta por blocos densos. Nestes blocos, as camadas estão densamente conectadas entre si: cada camada recebe como entrada os mapas de características de saída de todas as camadas anteriores. O DenseNet-121 é composto por 4 blocos densos, que, por sua vez, incluem de 6 a 24 camadas densas.

![image](https://github.com/RafaelGallo/Coursera_IA_medicina/assets/44452165/7cf675db-cd9a-4ef9-89e2-b5b5d07d91eb)


Bloco Densos: Um bloco denso é formado por n camadas densas. Essas camadas densas estão conectadas de forma que cada camada densa recebe mapas de características de todas as camadas anteriores e repassa seus mapas de características para todas as camadas subsequentes. As dimensões das características (largura e altura) permanecem iguais em um bloco denso.

![image](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/img/dense_block.png?raw=true)

Camada Densa: Cada camada densa consiste em 2 operações de convolução:

1 X 1 CONV (operação convencional de convolução para extração de características)

3 X 3 CONV (diminuição da profundidade/número de canais das características)

![image](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/img/dense_layer.png?raw=true)

## Configurações do modelo

A camada CONV corresponde à sequência BatchNorm-> ReLU->Conv. Cada camada tem essa sequência repetida duas vezes, a primeira com convolução 1x1 produzindo mapas de características com uma taxa de crescimento x 4, a segunda com convolução 3x3. Os autores descobriram que o modo de pré-ativação (BN e ReLU antes da convolução) era mais eficiente do que o modo de pós-ativação usual.

![image](https://github.com/RafaelGallo/Coursera_IA_medicina/assets/44452165/3fe4ee0a-6a5e-41c6-9a4a-2e2949fbafde)

A taxa de crescimento (k = 32 para o DenseNet-121) define o número de mapas de características de saída de uma camada. Basicamente, as camadas geram 32 mapas de características que são adicionados a 32 mapas de características das camadas anteriores. Enquanto a profundidade aumenta continuamente, cada camada traz de volta a profundidade para 32.

Camada de Transição: Entre os blocos densos, você encontra uma camada de transição. Em vez de somar o residual como na ResNet, o DenseNet concatena todos os mapas de características. Uma camada de transição é composta por: Normalização em Lote (Batch Normalization) -> Convolução 1x1 -> Pooling Médio (Average Pooling). 

As camadas de transição entre dois blocos densos garantem o papel de down-sampling (diminuição das dimensões x e y pela metade), essencial para redes neurais convolucionais (CNN). As camadas de transição também comprimem o mapa de características e reduzem pela metade os canais. Isso contribui para a compacidade da rede.
Embora a concatenação gere muitos canais de entrada, a convolução do DenseNet gera um baixo número de mapas de características (os autores recomendam 32 para um desempenho ideal, mas alcançaram desempenho de nível mundial com apenas 12 canais de saída).

## Principais benefícios

Compacidade. O DenseNet-201 com 20 milhões de parâmetros produz um erro de validação semelhante a uma ResNet de 101 camadas com 45 milhões de parâmetros.
As características aprendidas não são redundantes, uma vez que são compartilhadas por meio de conhecimento comum.
Facilidade de treinamento, porque o gradiente flui de volta mais facilmente devido às conexões curtas.
Em relação às configurações do modelo, neste projeto, o modelo utiliza imagens de raios-X de 320 x 320 pixels e gera previsões para cada uma das 14 patologias, como ilustrado em uma imagem de exemplo.

## Ambiente e dependências

Para executar o modelo, foi usado um ambiente com TensorFlow 1.15.0 e Keras 2.1.6. Os pesos do modelo estão incluídos no repositório.

## Resultados

Foi utilizado um modelo pré-treinado cujo desempenho pode ser avaliado usando a curva ROC mostrada abaixo. Os melhores resultados foram alcançados para Cardiomegalia (AUC de 0,9), Edema (0,86) e Massa (0,82). Idealmente, desejamos estar significativamente mais próximos de 1. Você pode conferir o desempenho do artigo ChexNeXt e seu modelo, assim como o desempenho de radiologistas nesse conjunto de dados.

Observando raios-X não vistos anteriormente, o modelo prevê corretamente a patologia predominante, gerando um diagnóstico relativamente preciso, destacando a região-chave que fundamenta suas previsões. Além do diagnóstico principal (com maior probabilidade), o modelo também prevê problemas secundários, semelhantes ao que um radiologista comentaria como parte de sua análise. Isso pode ser resultado de falsos positivos decorrentes de ruído nas radiografias ou de patologias cumulativas.

![img](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/img/result1.png?raw=true)

O modelo prevê corretamente Cardiomegalia e ausência de massa ou edema. A probabilidade de massa é maior, e podemos ver que pode ser influenciada pelas formas no meio da cavidade torácica, bem como ao redor do ombro.

![img](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/img/result2.png?raw=true)

Aqui, o modelo identifica corretamente os sinais de edema na parte inferior da cavidade torácica. Também podemos notar que a Cardiomegalia possui uma pontuação alta para esta imagem, embora o diagnóstico real não a inclua. Essa visualização pode ser útil para a análise de erros; por exemplo, podemos observar que o modelo está de fato observando a área esperada para fazer a previsão.

![image](https://github.com/RafaelGallo/Coursera_IA_medicina/assets/44452165/080c7d13-365d-4fec-adb0-887755695073)


Desempenho do artigo ChexNeXt (modelo e também radiologistas)

## Projeto 1 Notebook
[Notebook](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/notebook/Curso_1_IA_para_diagn%C3%B3stico_m%C3%A9dico/Projeto_01/C1M1_Assignment.ipynb)

## Projeto 2 Notebook
[Notebook](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/notebook/Curso_1_IA_para_diagn%C3%B3stico_m%C3%A9dico/Projeto_02/C1M2_Assignment.ipynb)

## Projeto 3 Notebook
[Notebook](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/notebook/Curso_1_IA_para_diagn%C3%B3stico_m%C3%A9dico/Projeto_03/C1M3_Assignment.ipynb)

![](https://github.com/RafaelGallo/Coursera_IA_medicina/blob/main/img/68747470733a2f2f6a6f75726e616c732e706c6f732e6f72672f706c6f736d65646963696e652f61727469636c652f6669677572652f696d6167653f73697a653d6c617267652669643d31302e313337312f6a6f75726e616c2e706d65642e313030323638362e74303031.png?raw=true)

## Conclusão 

Este projeto faz parte de um curso de IA para Medicina, onde estamos desenvolvendo uma rede neural convolucional (CNN) voltada para a classificação de imagens médicas. A arquitetura do modelo DenseNet, que foi previamente treinado, se destacou ao apresentar excelentes resultados na análise de imagens médicas. Essa abordagem inovadora promete um impacto significativo na precisão dos diagnósticos, destacando-se como uma ferramenta valiosa na área da medicina. Combinando a eficiência do DenseNet com a expertise médica, este projeto tem o potencial de aprimorar a prática médica e melhorar o atendimento ao paciente.

# Citação
Esse projeto na pratica da formação IA para medicina @DeepLearning.AI 
