# Mathematics-used-in-AI

이것의 내용은 wiki부분에 있습니다.

# t-test,ANOVA

## null hypothsis,alternative hypothesis
우선  t-test를 하기 전에 null hypothsis와 alternative hypothesis를 알아야한다.

null hypothsis(귀무가설)는 통계학에서 처음부터 버릴 것을 가정한 가설이고 alternative hypothesis(대립가설) 통계학에서 처음부터 채택할 것을 가정한 가설이다.  
ex)  
null hypothsis:어느 집단의 평균 m=30이다. alternative hypothesis: 어느 집단의 평균은 m=30이 아니다. 여기서 p(유의확률 0.05,0.01,0.001)보다 크면 (null hypothsis)귀무가설을 채택하고 작으면 (alternative hypothesis)대립가설을 채택한다.  
이렇게 처음 부터 대립가설을 세우지 않고 귀무 가설을 세우는 이유는 참이라고 증명하는 것보다 참이 아니라고 증명하는 것이 통상적으로 더 쉽고 귀무 가설을 올바르게 세우는 것이 대립가설을 정확하게 세우는 것 보다 더 쉽기 때문이다.

## t-test
t-test는 두 집단 간 평균의 차이를 평가하는 기법이다.

t-test에선 이 null hypothsis와 alternative hypothesis를 사용한다. 
null hypothsis: 두 집단의 평균이 같다. alternative hypothesis:두 집단의 평균이 같지 않다.  
여기서 p(유의확률)dl 0.05보다 작으면 null hypothsis가 버려지고 alternative hypothesis가 채택된다.

t-test에는 One-tailed(단측 검정)과 two-tailed(양측 검정)으로 나눠진다. One-tailed은 "크다.", "작다."를 검정하는 방법이고 two-tailed는 "같다.", "작지 않다."를 검정하는 방법이다. 양측 검정이 더 큰 범위이다. 따라서 데이터에 대해 잘 모르면 양측 검정을 사용하는 것이 낫다.  
또한 t-test에는 paired t-test와 independent t-test라는 것이 있다. paired t-test는 동일한 표본을 대상으로 처치(treatment) 전후의 효과를 검정하는 것이고 independent t-test는 독립적인 두 집단 간의 평균 차이를 검정하는 것이다.  
ex)
paired t-test:A를 먼저 테스트하고 그 다음 B를 테스트할 사용자 30명을 모집한다.  
independent t-test:A를 테스트할 사용자 30명과 B를 테스트할 사용자를 각각 모집한다.

## ANOVA(ANalysis of Variance)
세 개 이상의 집단에 평균 차이를 평가하는 기법이다.

null hypothsis: 세 집단의 평균이 같다. alternative hypothesis:세 집단의 평균이 같지 않다.  
여기서 "세 집단의 평균이 같지 않다."는 세 집단의 평균이 다르다는 것이 아니다. 즉 A!=B=C or A=B!=C or A!=B!=C 이라는 뜻다. 또한 post hoc-test라는 것이 있는데 이것은 집단을 2개씩 묶어서 비교한 것이다.

ANOVA는 크게 4가지의 종류가 있다. One-way ANOVA,two-way ANOVA,multi-variate ANOVA  
One-way ANOVA는 n개 집단을 1개를 분석할 때 사용하고 two-way ANOVA은 2개, multi-variate ANOVA은 3개이상일 때 사용한다. ex) "ABC의 집단에서 집단별로 키의 차이가 있는가? 또한 성별에 따른 차이가 있는가?--two-way ANOVA  
추가로 다변량 분산 분석이라는 것이 있는데 이것은 종속변수가 2개이상 인것을 의미한다. ex) "A,B,C의 특성 x1뿐만 아니라 특성 x2도 분석한다."

## t-test와 ANOVA를 사용할 수 있는 조건
### Normality test(정규성 검정)
 집단 간의 차이를 검증할 때 주로 사용되는 t-test나 ANOVA와 같은 통계 방법은 데이터가 정규 분포를 따른 다는 것을 가정하고 만들어졌다. 따라서 데이터가 정규 분포를 따라야한다. 정규 분포를 따르지 않는다고 판단되면 비모수적 검정을 사용해야한다.

모수적 검정: 모집단의 분포 형태가 정규 분포라고 가정(t-test,ANOVA가 해당), 비모수적 검정:모집단이 특정 분포라고 가정할 수 없거나 표본이 너무 적을 때 사용(wilcoxon signed rank test, kruskai-walls test가 여기에 해당)

### Homogeneity of variance(분산의 동질성)
 집단 간의 분산이 동일해야한다.  집단간의 분산이 다르다면 비교의 기준이 흔들려서 분산분석의 신뢰도가 나빠지게 된다. 이 분산 동실성을 검정하기 위한 도구로 Levene's test(레빈 검정)이라는 것이 있다.

 만약 분산이 다르다면 t-test의 경우 비모수적 검정인 Mann Whitney U test를 사용해야하고 ANOVA는 Kruskal-Walls를 사용해야한다.

 또한 집단간의 분산이 다르다면 그에 대한 조치로 정규화나 표준화를 시도해 볼 수 있다.(위의 방식을 사용하는 것보다 이것을 먼저해야한다.)

 ### independent observations(데이터 독립성)
  A그룹과 B그룹이 독립이어야한다. 즉, A그룹이 증가하면 B그룹도 증가하는 경향을 보이면 안된다.

# Normal Distribution(정규 분포)
정규 분포는 연속적으로 발생하는 사건의 확률 분포이다.  
표기로는 이렇게 나타낸다 N(x:m,s^2) 즉 확률변수 x는 평균=m, 분산=s^의 정규분포를 따른다는 뜻이다.  
정규 분포 수식은 이러하다. ![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/ae47f4a4-b091-44f5-802c-71bee013e0e0)  
여기서 1/N은 면적의 합을 1로 만들어주는 스케일링 상수다.

정규 분포는 probablity denstiy function(pdf)을 표현하는데 필요한 무한개의 데이터를 평균과 분산만으로 표현할 수 있다.

## standard univariate normal distribution(표준 정규 분포)
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/bd7706f1-357e-42bb-b246-71d39c1b90ed)  
표준 정규 분포는 분산을 1 평균을 0으로 만든 정규 분포이다.

univariate normal distribution는 표준 정규 분포에서 평균만큼 shifting하고 분산만큼 scaling 한것이다.  
x=s(scaling)z+m(shifting)

## Multivariate Gaussian Models
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/c931a5d9-5e8f-4c7d-8348-9b644ca143a1)

 다변수 가우시안 모델은 정규 분포를 다차원 공간(하나의 변수가 아닌 여러개의 변수)으로 확장한 것으로 행렬형태로 표현된다.
 
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/8cf6463f-4bc9-4e9f-9b53-a5dbc0775895)  
확률 변수가 2개일때 이 부분은 타원 방정식으로 생각할 수 있다.

![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/c922bff1-b661-4ca0-b9d9-b929db78ad55)

![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/ccdc523e-0703-4646-b7a6-21316f4096d2)  
저렇게 나온 타원은 사진에서의 단면을 의미한다.

# Bayes' theorem(베이즈 정리)
베이즈 정리는 두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 정리이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/0c0f270f-03c4-4161-ab64-391f3d2b25a5)
 여기서:H는 Hypothesis로 사전확률을 의미하고 E는 Envidence로 증거를 의미한다. 즉 증거를 이용하여 사전확률을 업데이트 한 사후확률을 구하겠다는 의미이다.

ex) 예를 들어 코로나에 걸릴 확률이 Hypothesis고 기침할 확률이 envidence라고 해보자. P(기침|코로나)=0.4이고 p(기침|코로나 아님)=0.3이고 P(코로나)=0.5(사전확률)이라 했을때 기침에 의해 업데이트된 p(코로나|기침)=p(기침|코로나)*p(코로나)/(p(기침|코로나)*p(코로나)+p(기침|코로나 아님)*p(코로나 아님))으로 0.5714가 된다. 즉 베이즈 정리는 불확식성을 의미 있는 데이터 기반으로 업데이트 해주는 정리라고 할 수 있다.

ex2) 1%의 발병률을 가진 코로나 바이러스가 있다. 또한 이 바이러스의 진단키트가 코로나에 걸렸을 때 양성으로 판정할 확률이 99%이고 아닌 경우 음성이라 판정할 확률은 98%이다. 이때 진단키트에서 양성 판정이 나왔을 떄 이 사람이 코로나일 확률을 구하고 두번째 검사에서도 양성일 때 코로나일 확률을 구하여라.  
이 문제에 대한 해답은 다음과 같다.

![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/583613e8-a4ef-47bb-ab72-7c3de871ac9c)

## 몬티홀 문제와 베이즈 정리
 몬티홀 문제는 다음과 같다.  
 "똑같이 생긴 문 3개 중에 하나의 문에만 자동차가 있고 나머지 문에는 염소가 있다. 우승자가 한개의 문을 선택했을 때 사회자는 우승자가 선택한 문을 제외한 두문 중 하나를 열어 염소 인것을 보여준다. 이때 우승자가 선택을 바꾸는 것이 유리한가? 아니면 바꾸지 않는것이 유리한가?"  
 이 문제의 답은 바꾸는 것이다. 이것을 베이즈 정리를 이용해 풀이해보겠다.  
 ![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/6ad196e0-1f6a-4c47-8c22-28bbe30631fc)

 인공지능에서는 이 베이즈 정리를 이용한 나이브 베이지안 분류기라는 것이 있다.

 # Maximum Likelihood Estimation(최대 가능도 추정)
 Maximum Likelihood Estimation(최대 가능도 추정)은 어떠한 확률 밀도 함수에서 관측된 표본데이터를 가지고 분산이나 평균같은 모수를 추정하는 것이다. 
 Likehood(우도는) 연속 확률 분포에서 특정 사건이 발생할 가능성으로 이산 확률 분포의 확률과는 다른 개념이다. 우도를 그림으로 표현하면 다음과 같다.  
 ![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/3b8e2a98-a46f-42c5-833a-22c6952c16ce)

Maximum Likelihood Estimation은 샘플 데이터(given data x)를 가장 잘 표현하는 분포를 구하는 것이다. 즉 그 분포의 평균과 분산을 구한다. 그럼 데이터에 최대한 맞는 분포를 찾으려면 어떻게 해야할까? 바로 주어진 데이터의 가능도(분포식에 x값을 넣은것)를 모두 곱한 값이 최대가 되는 분산과 평균을 찾으면 된다.  
ex) {68,69,70,71,72} 이 표본 데이터를 활용해 최대 가능도 추정으로 평균을 구해보시오.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/afc1ffe3-7d79-49f6-b3e0-b229cc800b22)  
위의 사진과 같이 먼저 가능드를 모두 곱해주고 미분을 편하게 하기위해 로그를 취해준다. 그 다음 가능도가 최대가 될떄의 m값을 구해야하니 m에 대해 미분한 값=0을 통해 m의 값을 추정할 수 있다.

# Principal Component Analysis, PCA((주성분 분석)
NOTICE:PCA을 이해하기 위해선 고유벡터의 개념을 알아야한다.  

Principal Component Analysis, PCA(주성분 분석)은 여러개의 독립변수들(고차원 데이터)의 특징을 잘 설명해 줄 수 있는 주성분을 추출해내는 기법이다. 이 주성분 분석은 고차원의 데이터를 저차원으로 투영시켜 학습시킬 때 자원을 아낄 수 있고 시각적으로 표현이 가능하게 해준다. 또한 주성분 분석을 통해 나온 특징들은 상관관계가 제거 되어 나온다. 즉 다중공신성을 제거해준다.
PCA의 기본적인 원리는 데이터들의 특징을 살리기 위해 분산이 최대가 축에 투영 시키는 것이다.(또는 투영 시킨 점과 기존데이터의 차이를 최소화하는 축을 찾는다.)  이 과정은 다음과 같다.  
먼저 데이터들의 평균이 0이되도록 shifting 해준다.(선형 변환을 위해선 필수적이다.) 그다음 분산이 1이되도록 scaling을 해준다.(단 분산이 중요한 데이터이면 하지 않아야한다.) 이후 변환시킨 데이터들을 투영했을 때 그 분산이 최대가 되는 단위 벡터를 찾는다. 이 과정을 수식적으로, 자세히 나태내면 아래 그림처럼 된다.

![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/d53535d6-e8f0-40ff-8b9f-ccfeeb946bd7)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/d9c9faea-e7dc-4f65-87ba-c776deace64c)  
위의 결과를 보면  투영된 값들의 분산값이 shifting and scaling된 데이터의 공분산의 고윳값이라는 것을 알 수 있다. 또한 고유벡터가 축의 벡터라는 것도 알 수 있다. 또한 n개의 특성을 가진 데이터에서 최대 n개의 주성분을 추출할 수 있다는 것도 알 수 있다.(고유벡터의 특징)  
+PCA로 추출된 주성분의 공분산 행렬을 구하면 이런 형태가 나온다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/072089a2-6877-481c-b0c9-aae722579a70)  
우선 z1,z2,z3끼리의 공분산은 주성분 추출과정에서 다중 공신성이 제거 되어 거의 0에 가깝게 나온다. z1의 분산은 주성분의 분산 즉 축에 투영된 데이터가 최대가 되는 분산이브로 원본(shifting scalinge된)데이터의 고윳값이 된다.

# Regression Analysis(회귀 분석)

## Linear Regression(선형 회귀)
인공지능에서 가장 일반적으로 쓰이는 회귀 분석 선형 함수를 사용하여 독립변수와 예측변수간의 관계를 도출해냄 행렬로 나타내면 아래와 같다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/401d1ec7-6761-4e64-a8a8-46e845f9b6b5)  
또한 회귀식이 p=wx+b일때 실제 값하고의 차이는 다음과 같다. error={(y-(wx+b)}^2 error가 최소가 되는 w와 b를 구하는 방법은 다음과 같다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/8056b4a7-e00c-4398-9b02-e0f9141a4359)  
위의 만들어진 1번과 2번식을 연립하면 w와 b를 구할 수 있다.

