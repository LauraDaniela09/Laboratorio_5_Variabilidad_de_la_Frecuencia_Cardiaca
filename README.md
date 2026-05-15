# Laboratorio-5-Frecuencia-cardiaca-HRV y balance atonomico
𝙞𝙣𝙩𝙧𝙤𝙙𝙪𝙘𝙘𝙞ó𝙣

La variabilidad frecuenciaacardíacadíaca (HRV) es un parámetro fisiológico que permite evaluar el equilibrio entre las ramas simpática y parasimpádela del sistema nervioso autónomo a través del análisis de los intervalos R-R obtenidos de la señal electrocardiográfica (ECG). Este parámetro es una herramienta fundamental en el estudio de la regulación cardíaca, ya que refleja la capacidad del corazón para adaptarse a diferentes estados fisiológicos, como el reposo o la actividad mental. En esta práctica se analizó la HRV mediante técnicaorganismodigital de señales, utilizando herramientas computacionales como Python para el filtrado, segmentación y análisis de los datos, con el fin de comprender cómo las variaciones en la frecuencia cardíaca pueden indicar cambios en la actividad autonómica del organismo.

𝙤𝙗𝙟𝙚𝙩𝙞𝙫𝙤

Identificar los cambios en el balance autonómico a partir del análisis temporal de la variabilidad de la frecuencia cardíaca (HRV), aplicando técnicas de procesamiento digital de señales para el filtrado y estudio de los intervalos R-R, y comparando la respuesta cardíaca en condiciones de reposo y durante la lectura en voz alta, con el propósito de relacionar la actividad simpática y parasimpática en ambos estados fisiológicos.

𝙞𝙢𝙥𝙤𝙧𝙩𝙖𝙘𝙞ó𝙣 𝙙𝙚 𝙡𝙞𝙗𝙧𝙚𝙧𝙞𝙖𝙨

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
```

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 A 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>


**PARTE A - A** 𝙛𝙪𝙣𝙙𝙖𝙢𝙚𝙣𝙩𝙤 𝙩𝙚𝙤𝙧𝙞𝙘𝙤 

𝟭. 𝘼𝙘𝙩𝙞𝙫𝙞𝙙𝙖𝙙 𝙎𝙞𝙢𝙥𝙖𝙩𝙞𝙘𝙖 𝙮 𝙋𝙖𝙧𝙖𝙨𝙞𝙢𝙥𝙖𝙩𝙞𝙘𝙖 𝙙𝙚𝙡 𝙨𝙞𝙨𝙩𝙚𝙢𝙖 𝙣𝙚𝙧𝙫𝙞𝙤𝙨𝙤 𝙖𝙪𝙩𝙤𝙣𝙤𝙢𝙤

El cuerpo está diseñado para poder mantener un equilibrio perfecto entre la actividad y el descanso ,esto es gracias al sistema nervioso autonomo, ya que este es el encargado de regular muchas de las funciones involuntarias que genera el organismo del cuerpo hconjuntoumano. 
dentro de este sistema se encuentra el sistema nervioso simpático y el sistema nervioso parasimpático los cuales son dos fuerzas opuestas encargadas de trabajar en  para poder mantener el bienestar del cuerpo.
<table>
  <tr>
    <td style="vertical-align: top; text-align: center;">
      <img src="https://github.com/user-attachments/assets/65cc303a-262d-4998-b7ef-1942768a875d" width="700"><br>
      <sub><b>imagen 1.</b> Sistema nervioso simpático.[1] </sub>
    </td>
    <td style="vertical-align: top; padding-left: 15px;">
sistema nervioso simpático: es el encargado de la aceleración de nuestro cuerpo, es decir es el responsable de la activación de la respuesta de huida cuando una persona se enfrenta a una situación de peligro o estrés .Es importante durante el reposo ya que es clave para preparar al cuerpo durante situaciones de emergencia .[2]
El sistema nervioso Simpatico funciona activando diversas vias donde se es evidente el aumento del ritmo cardiaco y respiratorio, presion sanguinea  dilatación de las pupilan,  cambios en el flujo sanguineo para que la sangre salga de la pie, estomago e intestinos para dirigirse hacia el cerebro,  corazón . y los diferentes musculos que sean  necesarios para llevar a cabo esta respuesta ante la actividad Simpatica.
    </td>
  </tr>
</table>



<table>
  <tr>
    <td style="vertical-align: top; text-align: center;">
      <img src="https://github.com/user-attachments/assets/bf0e4117-4f84-4f34-88ca-8376eb56190e" width="400"><br>
      <sub><b>imagen 2.</b> Sistema nervioso parasimpático.[1] </sub>
    </td>
    <td style="vertical-align: top; padding-left: 15px;">
Sistema nervioso  parasimpatico: controla la actividad de los musculo liso cardiaco y las glandulas. Es el  encargado de la respuesta  de descanso  debido a que esta involucrado en relentizar el  ritmo cardiaco , relajar los hemisterios en el tracto gastrointestinal y urinario y aumentar la actividad glandular e  intestinal.[3]como resultado se encuentra que el sistema parasimpatico es el encargado del almacenamiento de energía y la regulación de las funciones del cuerpo  como la digestion y la micción.
    </td>
  </tr>
</table>



𝟮. 𝙀𝙛𝙚𝙘𝙩𝙤 𝙙𝙚 𝙡𝙖 𝙖𝙘𝙩𝙞𝙫𝙞𝙙𝙖𝙙 𝙨𝙞𝙢𝙥𝙖𝙩𝙞𝙘𝙖 𝙙𝙚𝙡 𝙨𝙞𝙨𝙩𝙚𝙢𝙖 𝙣𝙚𝙧𝙫𝙞𝙤𝙨𝙤 𝙖𝙪𝙩𝙤𝙣𝙤𝙢𝙤:

La regulación de la frecuencia cardiaca está en manos de los sistemas nerviosos simpático y parasimpático. Ambos sistemas moderan la actividad de los nodos del corazón, así como la contracción miocárdica, todo esto por medio de comunicación electrónica y neuroquímica. 
El simpático va a favor de aumentar la frecuencia cardiaca, mientras que el parasimpático relaja el corazón y disminuye su bombeo. El balance entre estos dos mantiene la homeostasis cardiovascular, y la alteración de ese equilibrio desencadena condiciones y patologías.

El corazón es inervado por ambas ramas del sistema nervioso autónomo a traves del plexo cardíaco, que rodea la base del corazón y los grandes vasos. La inervación simpática se origina en la médula espinal a nivel torácico, las fibras preganglionares llegan al plexo y se distribuyen en los nodos SA, AV y el miocardio. El simpático libera noradrenalina en los receptores beta 1-adrenérgicos, generando así un aumento en  la contractilidad y frecuencia cardiaca. 

Por otro lado el sistema parasimpatico proviene del nervio vago, sus fibras preganglionares hacen sinapsis en los ganglios intrínsecos que estan ubicados en unas zonas grasas cardiacas y la pared auricular. Despues se libera acetilcolina ACh en los receptores muscarinicos M2 acoplados a proteinas G en el miocardio y los nodos. Esto resulta en la apertura de canales de potasio, hiperpolarizando la membrana del nodo SA, alejando el potencial de membrana del umbral que debe cumplir. Además disminuye los niveles de AMPc, lo cual disminuye la velocidad de conducción, realentizando la despolarización espontanea y por consecuente la contractilidad auricular y frecuencia cardiaca.    <sub><b></b> An overview of heart rate variability metrics and norms.[11] </sub>


𝟯. 𝙫𝙖𝙧𝙞𝙖𝙗𝙞𝙡𝙞𝙙𝙖𝙙 𝙙𝙚 𝙡𝙖 𝙛𝙧𝙚𝙘𝙪𝙚𝙣𝙘𝙞𝙖 𝙘𝙖𝙧𝙙𝙞𝙖𝙘𝙖 (𝙃𝙍𝘾) 𝙤𝙗𝙩𝙚𝙣𝙞𝙙𝙖 𝙖 𝙥𝙖𝙧𝙩𝙞𝙧 𝙙𝙚 𝙡𝙖 𝙨𝙚𝙣̃𝙖𝙡 𝙚𝙡𝙚𝙘𝙩𝙧𝙤𝙘𝙖𝙧𝙙𝙞𝙤𝙜𝙧𝙖𝙛𝙞𝙘𝙖 (𝙀𝘾𝙂).

La variabilidad de la frecuencia cardíaca (HRV) es un indicador fisiológico que mide las fluctuaciones en el intervalo de tiempo entre latidos consecutivos del corazón, conocidos como intervalos RR. Estos valores se obtienen a través de la señal electrocardiográfica (ECG). Este análisis proporciona información valiosa sobre el equilibrio entre las ramas simpática y parasimpática del sistema nervioso autónomo, responsables de regular la actividad cardíaca (Shaffer y Ginsberg, 2017).

Para calcular la HRV, se registra inicialmente la señal ECG y se identifican los complejos QRS, destacando los picos R, que caracterizan cada ciclo cardíaco. A continuación, se mide el tiempo entre dos picos R consecutivos (intervalo RR). Con esta serie de intervalos, se realiza el análisis de su variabilidad utilizando métodos en el dominio del tiempo o en el dominio de la frecuencia (Malik et al., 1996).

En el dominio del tiempo, los parámetros más habituales son la SDNN (desviación estándar de los intervalos RR) y la RMSSD (raíz cuadrada de la media de las diferencias cuadráticas sucesivas), que reflejan respectivamente la variabilidad global y la influencia del tono vagal. En el dominio de la frecuencia, se analizan componentes espectrales como LF (Low Frequency) y HF (High Frequency), que permiten estudiar el balance entre la actividad simpática y parasimpática. El cociente LF/HF se utiliza como indicador del equilibrio autonómico (Task Force, 1996).

La HRV se ha establecido como una herramienta valiosa tanto en entornos clínicos como en investigaciones. Una alta variabilidad suele asociarse con un sistema cardíaco saludable y una buena capacidad de adaptación fisiológica, mientras que una baja variabilidad puede ser señal de estrés, fatiga o disfunción autonómica. Por esta razón, el análisis de HRV a partir de la señal ECG es considerado un método no invasivo y fiable para evaluar la modulación autonómica del corazón (Shaffer y Ginsberg, 2017).

𝟰. 𝘿𝙞𝙖𝙜𝙧𝙖𝙢𝙖 𝙙𝙚 𝙥𝙤𝙞𝙣𝙘𝙖𝙧𝙚 𝙘𝙤𝙢𝙤 𝙝𝙚𝙧𝙧𝙖𝙢𝙞𝙚𝙣𝙩𝙖  𝙙𝙚 𝙖𝙣𝙖𝙡𝙞𝙨𝙞𝙨 𝙙𝙚 𝙡𝙖 𝙎𝙚𝙣̃𝙖𝙡 𝙍-

<table>
  <tr>
    <td style="vertical-align: top; text-align: center;">
      <img src="https://github.com/user-attachments/assets/9f20fe07-f172-4b1f-92ca-b875e5adee59" width="900"><br>
      <sub><b>imagen 3.</b> diagrama de poincare.[5] </sub>
    </td>
    <td style="vertical-align: top; padding-right: 15px;">
Es una herramienta para el analisis no lineal de la variable de la frecuencia cardiaca , obtenida a partir de la serie R-R , la cual representa los intervalos  de tiempo entre latidos consecutivos del corazón. Estos permiten mediante una transformación matematica representar graficamente la dinamica de la señal cardiaca. En este metodo , cada intervalo R-R se representan en  función del siguiente (RRn+1) generando una serie de puntos en el plano en forma de nube. la forma y dispersion refleja las caracteristicas  dinámicas del sistema cardiovascular permitiendo identificar patronos de regularidad, variabilidad o comportamiento. Este diagrama se presenta de manera bidimencional permitiendo simplificar la dinamica temporal de la señal (como varia la señal a lo largo del tiempo - intervalos entrelatidos de un momento a otro) y visualizar con claridad los patrones generales del comportamiento autonomico.[4][6]
    </td>
  </tr>
</table>

<table>
  <tr>
    <td style="vertical-align: top; padding-left: 15px;">
     Cuando se obtiene una figura en forma de elipse estrecha y alargada, se interpreta como baja variabilidad Esto implica que los intervalos entre latidos cambia muy poco, lo que refleja una menor flexibilidad del sistema nervioso autonomo para adaptarse  a las demandas internas y externas.
Por otro lado una figura más amplia , dispersa  o cercana a una forma  circular indica una alta variabilidad cardiaca  y un mejor equilibrio entre los sistemas Simpatico y parasimpatico. 
En comparación con otros metodos lineales, el diagrama de poincare  ofrece una vision topologica y geometrica de la dinamica cardiaca, permitiendo captar comportamientos no lineales que no se evidenciar mediante estadisticas convencionales. Por ello, esta herramienta se utiliza frecuentemente en el estudio de la serie R-R  para evaluar el control autonomico del corazón en condicion de repaso  estres o enfermedad.[6][7]
    </td>
    <td style="vertical-align: top; text-align: center;">
      <img src="https://github.com/user-attachments/assets/e76da245-3f4a-47bf-9c52-f9a9659ae756" width="1500"><br>
      <sub><b>imagen 4.</b> diagrama de poincare del corazon [8]</sub>
    </td>
  </tr>
</table>

**Ejemplos Relacionados**

| **Tipo de Ritmo**                 | **Descripción Breve**                                                                                           | **Imagen**                                                                                         |
|------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Ritmo Regular**                  | Intervalos constantes entre latidos, sin variación significativa.                                                | <img width="250" height="300" alt="Ritmo Regular" src="https://github.com/user-attachments/assets/b7ea282d-1fea-4d44-bc72-a54c5e601a19"/> <br> <sub><b>Imagen 5.</b> Diagrama y gráficas en ritmos regular (normal) [9]</sub>|
| **Ritmo Regular con Extrasístoles**| Latidos adicionales ocasionan irregularidades en los intervalos, visibles en el tacograma y diagrama de Poincaré.  | <img width="250" height="300" alt="Ritmo con Extrasístoles" src="https://github.com/user-attachments/assets/bee8d739-42df-4aa0-8e11-220593a7bb77"/> <br> <sub><b>Imagen 6.</b> Diagrama y gráficas en ritmo regular latidos ectópicos [9]</sub>|
| **Fibrilación Auricular**          | Patrones caóticos e irregulares, con intervalos muy variables y dispersión de puntos en el diagrama de Poincaré. | <img width="250" height="300" alt="Fibrilación Auricular" src="https://github.com/user-attachments/assets/4d3436bc-ed4f-4ebc-860f-13a8c78edac2"/> <br> <sub><b>Imagen 7.</b> Diagrama y gráficas en fibrilación auricular [9]</sub>|



**variabilidad e intereptacion del diagrama de poincare**

<div align="center">

<table style="width: 70%; text-align: center;">
  <tr>
    <th>Forma del Diagrama</th>
    <th>Variabilidad</th>
    <th>Sistema Nervioso Involucrado</th>
    <th>Interpretación</th>
  </tr>

  <tr>
    <td>Elipse estrecha y alargada</td>
    <td>Baja variabilidad</td>
    <td>Predomina el simpático</td>
    <td>Menor flexibilidad del corazón, poca adaptación.</td>
  </tr>

  <tr>
    <td>Elipse amplia</td>
    <td>Variabilidad moderada</td>
    <td>Equilibrio simpático–parasimpático</td>
    <td>Buena respuesta, aunque con cierta rigidez.</td>
  </tr>

  <tr>
    <td>Forma circular o dispersa</td>
    <td>Alta variabilidad</td>
    <td>Fuerte equilibrio autónomo</td>
    <td>El corazón se adapta muy bien a los cambios.</td>
  </tr>

  <tr>
    <td>Puntos irregulares</td>
    <td>Variabilidad baja o irregular</td>
    <td>Desbalance autónomo</td>
    <td>Relacionado con estrés o posibles alteraciones.</td>
  </tr>

</table>

</div>
A su vez tambien se obtienen los parámetros SD1 y SD2. El primero está asociado a las diferencias entre intervalos RR consecutivos. El segundo está relacionado con la desviación estándar de los intervalos RR.

| **Parámetro** | **Descripción**                                          | **Relación con la variabilidad cardíaca**                     | **Significado en términos del sistema nervioso autónomo**                 |
|---------------|----------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------|
| **SD1**       | Desviación estándar en la dirección perpendicular a la línea de identidad. | Asociado a la variabilidad a corto plazo (fluctuaciones rápidas). | Refleja la **actividad parasimpática** (control del sistema nervioso autónomo que regula el descanso y la recuperación). |
| **SD2**       | Desviación estándar en la dirección paralela a la línea de identidad. | Asociado a la variabilidad a largo plazo (fluctuaciones lentas). | Refleja el **equilibrio entre los sistemas simpático y parasimpático**. |


**Plan de acccion** 

A continuación, se presentará el plan de acción que se implementará para abordar los puntos clave del análisis. Este plan tiene como objetivo facilitar la comprensión de los resultados y definir las siguientes acciones a seguir.

<p align="center">
  <img src="https://github.com/user-attachments/assets/33e4c00e-feb1-4874-a904-b7d704122998" width="300">
</p>

**PARTE A - B**
Adquisición de la señal ECG
Lo primero que se hizo fue adquirir la señal atravez del bitalino luego de tener la señal se realizo un filtro IIR para filtrar a señal original.

**Valores del filtro**

$$
y[n]=0.22x[n]+0.66x[n-1]+0.66x[n-2]+0.22x[n-3]+0.66y[n-1]-0.11y[n-2]
$$

<img width="449" height="148" alt="image" src="https://github.com/user-attachments/assets/688e83b8-0284-4635-9b14-7be54f96203c" />

```python
plt.figure(figsize=(15,4))

plt.plot(t,
         ecg_mV,
         color='#C8A2FF')

plt.title("ECG Original")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.grid()

plt.show()

# =========================================================
# FILTRO IIR
# =========================================================

y = np.zeros(len(ecg_mV))

for n in range(3, len(ecg_mV)):

    y[n] = (
        0.22 * ecg_mV[n]
        + 0.66 * ecg_mV[n-1]
        + 0.66 * ecg_mV[n-2]
        + 0.22 * ecg_mV[n-3]
        + 0.66 * y[n-1]
        - 0.11 * y[n-2]
    )

# =========================================================
# ECG FILTRADO
# =========================================================

plt.figure(figsize=(15,4))

plt.plot(t,
         ecg_mV,
         label='Original',
         alpha=0.4,
         color='#D8BFD8')

plt.plot(t,
         y,
         label='Filtrada',
         color='#B57EDC',
         linewidth=2)

plt.title("ECG Filtrada")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.legend()

plt.grid()

plt.show()
```
Los coeficientes bk corresponden al numerador de la función de transferencia y afectan las muestras actuales y pasadas de la señal de entrada. Por otro lado, los coeficientes ak
	​

 provienen del denominador y están asociados a las salidas anteriores del sistema, generando retroalimentación. A partir de estos coeficientes se obtuvo la ecuación en diferencias del filtro
**Resultado**
<img width="1258" height="393" alt="image" src="https://github.com/user-attachments/assets/7ff244ad-e759-41b9-8a05-97105d470b16" />
se hizo una comparacion entre la señal original y la filtrada teniendo como resultado:
<img width="1258" height="394" alt="image" src="https://github.com/user-attachments/assets/74670f19-8767-469a-8dee-04d0dba62acb" />

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 B 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

teniendo la señal filtrada se corto en dos partes, la primera va del minuto 0 al minuto 2 el cual se concntra en que el paciente esta en completo silencio, sin movimiento.

**Resultado**
<img width="1258" height="393" alt="image" src="https://github.com/user-attachments/assets/12b28208-272a-446d-ab13-afdfd874040a" />

```python
muestras_2min = 120 * Fs

reposo = y[:muestras_2min]

lectura = y[muestras_2min:2*muestras_2min]

# =========================================================
# TIEMPOS
# =========================================================

t_reposo = np.arange(len(reposo))/Fs

t_lectura = np.arange(len(lectura))/Fs

# =========================================================
# ECG REPOSO
# =========================================================

plt.figure(figsize=(15,4))

plt.plot(t_reposo,
         reposo,
         color='#B57EDC')

plt.title("ECG Filtrado - Reposo")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.grid()

plt.show()
```

por otro lado esta del minuto 2 al minuto 4 que se encuentra en actividad, donde el paciente esta leyendo en voz alta durante los dos minutos.

**Resultado**
<img width="1258" height="393" alt="image" src="https://github.com/user-attachments/assets/8fde445e-99e8-4d27-8986-88d3ce83a2f6" />

```python
plt.figure(figsize=(15,4))

plt.plot(t_lectura,
         lectura,
         color='#B57EDC')

plt.title("ECG Filtrado - Lectura")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.grid()

plt.show()
```
luego se identificar los picos R en cada uno de los segmentos, calcular los intervalos R-R y obtener una nueva señal con dicha información.

  **PICOS R EN REPOSO**

<img width="1258" height="394" alt="image" src="https://github.com/user-attachments/assets/29eb5ac8-21b5-4edf-9847-848e74e0de7e" />

```python
DETECCIÓN DE PICOS R
# =========================================================

picos_reposo, _ = find_peaks(
    reposo,
    distance=Fs*0.5,
    prominence=0.1
)

picos_lectura, _ = find_peaks(
    lectura,
    distance=Fs*0.5,
    prominence=0.1
)

# =========================================================
# PICOS R - REPOSO
# =========================================================

plt.figure(figsize=(15,4))

plt.plot(t_reposo,
         reposo,
         color='#B57EDC')

plt.plot(picos_reposo/Fs,
         reposo[picos_reposo],
         "o",
         color='#6A0DAD',
         markersize=6)

plt.title("Detección de Picos R - Reposo")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.grid()

plt.show()
```
A partir de la detección de los picos R se calcularon los intervalos R-R consecutivos, generando una nueva serie temporal correspondiente a la variabilidad de la frecuencia cardíaca (HRV). Esta señal permite analizar las fluctuaciones del ritmo cardíaco bajo diferentes condiciones fisiológicas.

<img width="1263" height="393" alt="image" src="https://github.com/user-attachments/assets/3633a4c5-e1e4-48dc-aff9-635be63f586e" />

**PICOS R EN ACTIVIDAD**

<img width="1258" height="393" alt="image" src="https://github.com/user-attachments/assets/dcc47b94-96bb-4fe2-a387-c6319f65fba1" />


```python
PICOS R - LECTURA
# =========================================================

plt.figure(figsize=(15,4))

plt.plot(t_lectura,
         lectura,
         color='#B57EDC')

plt.plot(picos_lectura/Fs,
         lectura[picos_lectura],
         "o",
         color='#6A0DAD',
         markersize=6)

plt.title("Detección de Picos R - Lectura")

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")

plt.grid()

plt.show()

# =========================================================
# INTERVALOS RR
# =========================================================

rr_reposo = np.diff(picos_reposo)/Fs

rr_lectura = np.diff(picos_lectura)/Fs

# =========================================================
# HRV
# =========================================================

print("========== REPOSO ==========")

print("Media RR:", np.mean(rr_reposo), "s")

print("SDNN:", np.std(rr_reposo), "s")

print("\n========== LECTURA ==========")

print("Media RR:", np.mean(rr_lectura), "s")

print("SDNN:", np.std(rr_lectura), "s")
```
<img width="1265" height="393" alt="image" src="https://github.com/user-attachments/assets/060a8f1c-8add-464a-88f7-c95dc24ef6ff" />

<img width="386" height="204" alt="image" src="https://github.com/user-attachments/assets/88b335b1-cdc1-417b-9287-f93c0b834e1a" />

La media de los intervalos R-R representa el tiempo promedio entre latidos cardíacos consecutivos, mientras que el parámetro SDNN corresponde a la desviación estándar de dichos intervalos y se utiliza como indicador de la variabilidad de la frecuencia cardíaca. Estos parámetros permiten evaluar la actividad del sistema nervioso autónomo bajo diferentes condiciones fisiológicas.

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 C 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

 **Construcción del diagrama de Poincaré**
 
Por ulimo ya teniendo los intervalos R-R se puede realizar el diagrama de poincaré tanto para la condición de reposo como para la condición de actividad (lectura). Cada punto del gráfico representa la relación entre dos intervalos cardíacos consecutivos.
 
 **Resultado del diagrama de poincaré en reposo**
 <img width="553" height="548" alt="image" src="https://github.com/user-attachments/assets/31f6bb09-59c1-4e4a-848b-2d709093096b" />

```python
plt.figure(figsize=(6,6))

plt.scatter(rr_reposo[:-1],
            rr_reposo[1:],
            color='#B57EDC')

plt.xlabel("RR(n) [s]")
plt.ylabel("RR(n+1) [s]")

plt.title("Diagrama de Poincaré - Reposo")

plt.grid()

plt.axis('equal')

plt.show()
```
En el estado de reposo, los puntos del diagrama tienden a concentrarse más cerca de la diagonal principal, indicando una mayor estabilidad en el ritmo cardíaco.

**Resultado del diagrama de poincaré en actividad**
<img width="536" height="548" alt="image" src="https://github.com/user-attachments/assets/5e7d54e1-9d8d-433a-b05c-ca29a99a73f9" />

```python
plt.figure(figsize=(6,6))

plt.scatter(rr_lectura[:-1],
            rr_lectura[1:],
            color='#B57EDC')

plt.xlabel("RR(n) [s]")
plt.ylabel("RR(n+1) [s]")

plt.title("Diagrama de Poincaré - Lectura")

plt.grid()

plt.axis('equal')

plt.show()
```
Durante la actividad se observó una dispersión diferente de los puntos, reflejando cambios en la variabilidad cardíaca asociados a la activación del sistema nervioso autónomo.

<h1 align="center"><i><b>Bibliografia</b></i></h1>
[1]Researchgate.net.de https://www.researchgate.net/figure/Figura-173-Los-sistemas-simpatico-y-parasimpatico_fig2_313160220

[2]Sistema nervioso simpático. (2023, 30 octubre). Kenhub. https://www.kenhub.com/es/library/anatomia-es/sistema-nervioso-simpatico

[3]Sistema nervioso parasimpático. (2023, 30 octubre). Kenhub. https://www.kenhub.com/es/library/anatomia-es/sistema-nervioso-parasimpatico

[4]Fishman, M., Jacono, F. J., Park, S., Jamasebi, R., Thungtong, A., Loparo, K. A., & Dick, T. E. (2012). A method for analyzing temporal patterns of variability of a time series from Poincaré plots. Journal Of Applied Physiology, 113(2), 297-306. https://doi.org/10.1152/japplphysiol.01377.2010

[5]Researchgate.net.de https://www.researchgate.net/figure/Figura-95-Diagramas-de-Poincare-y-parametros-tipicos-calculados-a-partir-de-la-senal-ECG_fig18_39569851

[6]Hrv_Admin. (s. f.). Understanding the Poincaré plot – HRV Health. https://hrvhealth.org/blog/?p=124

[7] CH González Obregón Upc.edu.(2002), de https://upcommons.upc.edu/server/api/core/bitstreams/eafd6950-b60c-462d-8eeb-1c6e1ee97891/content

[8]Researchgate.net.de https://www.researchgate.net/figure/Figura-94-Diagrama-de-Poincare-de-una-serie-RR_fig17_39569851

[9]Academia FibriCheck(2023) de https://academy.fibricheck.com/hc/en-be/articles/9020344305564-Chapter-8-Differentiating-regular-and-irregular-PPG-recordings

[10]Malik, M., et al. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. Circulation, 93(5), 1043–1065.

[11]Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in Public Health, 5, 258.

[12]Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: Standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043–1065.
