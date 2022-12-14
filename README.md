# BA_Anomaly_Detection
## 1. Anomaly Detection
이상탐지(Anomaly Detection)이란, 데이터에서 다른 패턴을 보이는 개체 또는 자료를 찾는 것을 의미한다. 즉 학습데이터를 기반으로 기존 데이터들과는 다른 특성을 갖는 데이터를 찾는 방법이다. 사이버 보안, 의학 분야, 금융 분야, 행동 패턴 분야 등 다양한 분야에 적용될 수 있다. 대표적인 예로 신용카드 사기, 사이버 침입, 테러 행위 같은 악의적 행동이나 시스템 고장, 비정상적인 상황등에 활용된다.
'이상' 이라는 표현은 적용되는 도메인 컨텍스트나 데이터의 종류에 따라 Anomaly, outlier, discordant observation, exception, aberration 등 다양하게 불린다.

### 1.2 Anomaly Detection vs Classification
이상 탐지는 종종 분류(Classification)문제와 혼동되는데, 둘은 엄연한 차이가 있다. 분류는 두 범주를 구분할 수 있는 경계면을 찾는 것인 반면, 이상 탐지는 다수의 범주를 고려해 이상치가 아닌 데이터들의 sector를 구분 짓는 것이라고 할 수 있다. 

![image](https://user-images.githubusercontent.com/71392868/202358197-6b8e5f7a-7858-45ed-b564-8a843d622874.png)

출처: https://datanetworkanalysis.github.io/2020/02/05/understanding_outlier1

이상 탐지를 위해서는 다양한 접근이 가능하다. 그 기법들은 살펴보면 크게 분류 기반, 밀도 기반, Nearest Neighnor(NN) 기반, 군집화 기반, 통계적 기법 등으로 나뉠 수 있다.

## 2. 분류 기반 이상 탐지
분류기를 주어진 특성 공간(Feature Space)에서 학습시킬 수 있는 가정을 전제로 한다. 라벨의 개수에 따라 one-class 또는 multi-class로 데이터를 학습시키고, class에 해당하지 않는 개체를 이상치로 처리한다. 대표적으로 오토인코더(Autoencoder), One-Class SVM 알고리즘들이 있다.
### 2.1 Autoencoder
오토인코더 신경망의 구조는 아래 그림과 같다. 입력층(Input layer)과 출력층(Output layer)의 노드 수가 같고, 하나 이상의 은닉층(Hidden layer)으로 구성된 인코더를 통해 입력 데이터를 압축하고, 디코더를 사용해 개체를 복원한다. 이렇게 복원했을 때 발생하는 복원오차가 크면 클수록 이상 개체라고 판단하는 방식이다. One-class, multi-class 문제의 구분 없이 모두 적용할 수 있다.
![image](https://user-images.githubusercontent.com/71392868/202358740-ab232695-c7f3-4df4-a92b-24d06144c4ce.png)

출처: https://www.mdpi.com/1424-8220/21/19/6679

### 2.2. SVM
Class Imbalance가 매우 심한 경우 정상 sample만 이용해서 모델을 학습하기도 하는데, 이 방식을 One-Class Classification(혹은 Semi-supervised Learning)이라 한다. 이 방법론의 핵심 아이디어는 정상 sample들을 둘러싸는 discriminative boundary를 설정하고, 이 boundary를 최대한 좁혀 boundary 밖에 있는 sample들을 모두 비정상으로 간주하는 것이다. One-Class SVM 이 One-Class Classification을 사용하는 대표적인 방법론으로 잘 알려져 있으며, 이 아이디어에서 확장해 Deep Learning을 기반으로 One-Class Classification 방법론을 사용하는 Deep SVDD 논문이 잘 알려져 있다. Deep SVDD의 구조는 아래와 같다. 

![image](https://user-images.githubusercontent.com/71392868/202359716-70d60582-ef24-482c-82f4-d343483348d4.png)

출처: https://www.cognex.com/ko-kr/blogs/deep-learning/research/anomaly-detection-overview-1-introduction-anomaly-detection

## 3. 밀도 기반 이상 탐지
정상 데이터의 데이터 분포를 사용하여 정상 상태의 분포를 추정한 뒤, 새로운 객체에 대하여 확률이 높으면 정상, 확률이 낮으면 비정상을 반환하는 방법론이다. 일반적으로 데이터의 분포를 설명하는 모수 모형(Parametric Model을 가정하며, 정규 분포로 추정을 할 때 몇 개의 가우시안 모델이 사용되었는가에 아래와 같이 Gaussian Density와 Mixture Gaussian Density로 분류될 수 있다.


![image](https://user-images.githubusercontent.com/71392868/202360037-3e6f4484-0904-40dc-9a0e-281320dc0cfa.png)

출처: https://velog.io/@euisuk-chung/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%B0%A8%EC%9B%90%EC%B6%95%EC%86%8C-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%83%90%EC%A7%80-%EA%B8%B0%EB%B2%95-%EB%B0%80%EB%8F%84%EA%B8%B0%EB%B0%98-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%83%90%EC%A7%80

### 3.1. Gaussian Density Estimation

가정
- 관측치들은 하나의 Gaussian으로부터 생성되었다.

![image](https://user-images.githubusercontent.com/71392868/202360189-d1ba9c7a-786f-4499-8c25-658d717950e1.png)

출처: https://velog.io/@euisuk-chung/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%B0%A8%EC%9B%90%EC%B6%95%EC%86%8C-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%83%90%EC%A7%80-%EA%B8%B0%EB%B2%95-%EB%B0%80%EB%8F%84%EA%B8%B0%EB%B0%98-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%83%90%EC%A7%80

장점
* 데이터의 범위에 민감하지 않다. (∵ 공분산 행렬은 측정 단위가 영향을 끼치지 않음)

* 분포를 추정한 학습데이터로부터 처음부터 rejection에 대한 1종 오류를 정의할 수 있다. (ex. 신뢰수준 95%)


* Formulation (Parameter estimation: μ, σ2)

![image](https://user-images.githubusercontent.com/71392868/202360503-f013d303-3c05-40dd-bdfb-81dfa279372a.png)

### 3.2. Mixture of Gaussian Density Estimation

가정
* 관측치들은 여러 개의 Gaussian들의 선형결합으로부터 생성되었다.

* 가우시안 결합 모델과 각각의 가우시안 모델(수식)

![image](https://user-images.githubusercontent.com/71392868/202360579-a572c5ea-51d3-40ab-b3ee-3db4900ad240.png)

### 3.2 Isolation Forest


Isolation forest는 기본적으로 데이터셋을 의사결정나무(Decision Tree) 형태로 표현해 정상값을 분리하기 위해서는 의사결정나무를 깊숙하게 타고 내려가야 하고, 반대로 이상값은 의사결정나무 상단부에서 분리할 수 있다는 것을 이용한다. 이 특성을 사용해 의사결정나무를 몇 회 타고 내려가야 분리되는가를 기준으로 정상과 이상을 분리한다.
랜덤포레스트가 의사결정나무를 여러번 반복하여 앙상블 하듯이, Isolation Forest는 iTree를 여러번 반복하여 앙상블 한다.

장점 
* 군집기반 이상탐지 알고리즘에 비해 계산량이 매우 적다. 
* 강건한(Robust)한 모델을 만들 수 있다. 
* 변수가 많은 데이터에서도 효율적으로 작동가능하다. 


![image](https://user-images.githubusercontent.com/71392868/202361965-6d5bbeb0-7371-4523-bd39-787c3160b4ff.png)

출처: https://velog.io/@vvakki_/Isolation-Forest-%EB%AF%B8%EC%99%84%EC%84%B1


## 4. 군집화 기반


세 번째는 군집화(Clustering) 기반 기법들이며 이는 군집의 중심(Cenrtroid) 중 가장 가까운 것과의 거리가 긴 것을 이상치로 본다. 먼저 군집화를 하고 개체가 포함된 군집의 중심과 개체 사이 거리를 이상점수로 놓는 것이다. 학습 데이터를 군집화하고 테스트 개체를 군집과 비교해 이상 점수를 얻는 식이므로 준지도 기법이라고 볼 수 있다. 대표적으로 K-Means 방법을 이용한다.
k번째로 가까운 개체와의 거리를 이용하는 경우, 이상 점수를 기준으로 정렬시킨 뒤 가장 큰 k개를 이상 개체로 보는 방법이다. 이상 점수는 가까운 k개의 개체와의 거리의 합으로 구하거나, 한 개체에서 일정 거리 이내에 있는 개체의 수를 세는방법 등이 있다. 

![image](https://user-images.githubusercontent.com/71392868/202360913-2fcd7e9d-c14d-4104-8018-82034e7b0932.png)

출처: https://datanetworkanalysis.github.io/2020/02/05/understanding_outlier1

![image](https://user-images.githubusercontent.com/71392868/202363949-9b252246-9ee4-407a-8540-93ccbb4f5db3.png)

출처: https://www.semanticscholar.org/paper/KNN-Based-Outlier-Detection-Algorithm-in-Large-Yang-Huang/6368120fc4e9c4ad1610b101a1c5f53100d711e0

## 5. 통계적 기반

통계적 기법의 근본적 원칙은 ‘이상값은 가정된 확률분포에서 생성되 지 않아 부분적으로, 또는 완전히 동떨어졌다고 여겨지는 관측값이다’(Anscombe & Guttman, 1960)라는 것이다. 그리고 ‘이상값은 확률 분포에서 낮은 영역에 나타난다’고 가정한다. 통계적 기법은 주어진 자료로 (보통 정상값의) 모형을 적합한 뒤 통계적 추론을 통해 새로운 개체가 그 모형을 따르는지를 판단한다. 검정 통 계량을 바탕으로 학습된 모형으로부터 생성되었을 확률이 낮은 개체를 이상값으로 본다. 모수적, 비모수적 기법 모두 적용할 수 있다. 모수적 기법은 분포의 꼴을 미리 알고 있다고 가정하고 모수를 추정하는 반면 (Eskin, 2000), 비모수적 기법은 일반적으로 분포에 대한 가정이 없다 (Desforges et al., 1998).

### 5.1. 모수적기법 
테스트 대상 데이터가 추정된 분포에서 생성되었다는 것(정상값)을 귀무가설로 한다. 이때 가설 검정에 사용한 검정 통계량을 이상 score로 활용할 수 있다. 
  
모수적 기법은 분포의 종류에 따라 아래와 같이 나눌 수 있다.

#### 5.1.1. 정규모형 기반
데이터가 정규모형에서 생성된 것으로 가정하고, 최대우도추정량(maximum likelihood estimator, MLE) 를 사용.

각 데이터와 추정된 평균값 사이의 거리가 “이상 score” 가 되고, 이상 score 의 경계를 정해서 이상값 여부를 결정한다.

거리의 정의와 경계를 구하는 방법들

* 상자그림

* Grubbs 검정

* Mahalanobis 거리

* Student t 검정

* Hotelling’s t 검정

* 카이제곱 검정

#### 5.1.2. 회귀모형 기반 
시계열 데이터에 적용하며, 데이터의 회귀모형을 적합한 뒤에 테스트 데이터와 회귀모형간의 잔차(residual)로 “이상 score” 를 구한다.

* Robust 회귀

* ARIMA 모형

### 5.2. 비모수적 기법

데이터가 특정 모형을 따른다는 가정을 하지 않는다.
비모수적 기법은 실제로 데이터가 특정 분포를 따른다는 가정이 성립하지 않을 때가 많기 때문에 현실적인 접근이 용이한 이점이 있다.

* 히스토그램 기반

* 커널 함수 기반


통계적 접근의 강점 및 약점
장점 
* 변동없는수학기초 
* 해석 용이
* 분포가 알려진 경우 높은 성능
단점  
* 분포가 알려지지 않은 경우 낮은 성능 
* 고차원 데이터의 경우 실제분포를 예측하기 어려움
* 이상치는 분포의 매개변수를 왜곡시킬수 있음 


# Reference

* Reference1 : https://velog.io/@vvakki_/Isolation-Forest-%EB%AF%B8%EC%99%84%EC%84%B1
* Reference2 : http://docs.iris.tools/manual/IRIS-Usecase/AnomalyDetection/AnomalyDetection_202009_v01.html
* Reference3 : https://nanunzoey.tistory.com/entry/%EC%9D%B4%EC%83%81-%ED%83%90%EC%A7%80Anomaly-Detection-%EA%B8%B0%EB%B2%95%EC%9D%98-%EC%A2%85%EB%A5%98
* Reference4 : https://www.cognex.com/ko-kr/blogs/deep-learning/research/anomaly-detection-overview-1-introduction-anomaly-detection
* Reference5 : https://datanetworkanalysis.github.io/2020/02/05/understanding_outlier1
* Reference6 : https://www.mdpi.com/1424-8220/21/19/6679
* Reference7 : https://datanetworkanalysis.github.io/2020/02/05/understanding_outlier1
* Reference8 : Desforges, M. J., Jacob, P. J., &Cooper, J. E. (1998). Applications of probability density estimation to the detection of abnormal conditions in engineering. Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science, 212(8), 687-703.
* Reference9 : Eskin, E.Anomaly detection over noisy data using learned probability distributions. Paper presented at the In Proceedings of the International Conference on Machine Learning





