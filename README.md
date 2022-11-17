# BA_Anomaly_Detection
## 1. Anomaly Detection
이상탐지(Anomaly Detection)이란, 데이터에서 다른 패턴을 보이는 개체 또는 자룔르 찾는 것을 의미한다. 즉 학습데이터를 기반으로 기존 데이터들과는 다른 특성을 갖는 데이터를 찾는 방법이다. 사이버 보안, 의학 분야, 금융 분야, 행동 패턴 분야 등 다양한 분야에 적용될 수 있다. 대표적인 예로 신용카드 사기, 사이버 침입, 테러 행위 같은 악의적 행동이나 시스템 고장, 비정상적인 상황등에 활용된다.
'이상' 이라는 표현은 적용되는 도메인 컨텍스트나 데이터의 종류에 따라 Anomaly, outlier, discordant observation, exception, aberration 등 다양하게 불린다.

### 1.2 Anomlay vs Classification
이상 탐지는 종종 분류(Classification)문제와 혼동되는데, 둘은 엄연한 차이가 있다. 분류는 두 범주를 구분할 수 있는 경계면을 찾는 것인 반면, 이상 탐지는 다수의 범주를 고려해 이상치가 아닌 데이터들의 sector를 구분 짓는 것이라고 할 수 있다. 

![image](https://user-images.githubusercontent.com/71392868/202358197-6b8e5f7a-7858-45ed-b564-8a843d622874.png)
