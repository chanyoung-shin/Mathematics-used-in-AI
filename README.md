# english.ver
# Mathematics-used-in-AI (Final Version)
---

## 1. Mean, Expected Value, Variance

### Mean (평균)

- The **mean** is the sum of all data points divided by the number of points.  
- Variants include arithmetic mean, harmonic mean, and geometric mean.

### Expected Value (기댓값)

- The **expected value** is the long-run average outcome of a probabilistic event.
- In large-sample scenarios (e.g., MNIST), “expectation” often emphasizes the probabilistic viewpoint.

### Variance (분산)

For a **population** of \(n\) data points:

$$
\sigma^2 \;=\; \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

For a **sample**:

$$
s^2 \;=\; \frac{1}{\,n-1\,} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

---

## 2. MSE, RMSE, MAE

### MSE (Mean Squared Error)

$$
\text{MSE} \;=\; \frac{1}{n}\,\sum_{i=1}^n \bigl(y_i - \hat{y}_i\bigr)^2
$$

- Commonly used loss function in regression.
- Quadratic form → differentiable everywhere.
- Very sensitive to outliers.

### RMSE (Root Mean Squared Error)

$$
\text{RMSE} \;=\; \sqrt{\text{MSE}}
$$

- Also used for regression.
- Slightly reduces extreme distortions compared to MSE.
- Often used as an **evaluation** metric.

### MAE (Mean Absolute Error)

$$
\text{MAE} \;=\; \frac{1}{n}\,\sum_{i=1}^n \bigl|\,y_i - \hat{y}_i\bigr|
$$

- Less sensitive to large outliers.
- Commonly used to **evaluate** regression models.

---

## 3. Covariance and Correlation

### Covariance (공분산)

For two variables \(X\) and \(Y\), the sample covariance is:

$$
\mathrm{Cov}(X, Y)
\;=\;
\frac{1}{\,n-1\,}\,\sum_{i=1}^n
\bigl(x_i - \bar{x}\bigr)\,\bigl(y_i - \bar{y}\bigr)
$$

- \(>0\) implies a positive linear relationship.
- \(<0\) implies a negative linear relationship.
- Near 0 implies no linear relationship.

### Covariance Matrix

For \(n\) variables \(\{x_1, \dots, x_n\}\):

$$
\Sigma \;=\;
\begin{bmatrix}
\mathrm{Var}(x_1)
& \mathrm{Cov}(x_1, x_2)
& \dots
& \mathrm{Cov}(x_1, x_n) \\
\mathrm{Cov}(x_2, x_1)
& \mathrm{Var}(x_2)
& \dots
& \mathrm{Cov}(x_2, x_n) \\
\vdots & \vdots & \ddots & \vdots \\
\mathrm{Cov}(x_n, x_1)
& \mathrm{Cov}(x_n, x_2)
& \dots
& \mathrm{Var}(x_n)
\end{bmatrix}
$$

### Correlation Coefficient (상관계수)

$$
\rho_{X,Y}
\;=\;
\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\,\mathrm{Var}(Y)}}
$$

---

## 4. Normalization vs. Standardization

### Normalization (정규화)

$$
x^\prime
\;=\;
\frac{\,x - \min(x)\,}{\,\max(x)\;-\;\min(x)\,}
$$

- Rescales data to [0,1].
- Preserves shape but unifies scale.

### Standardization (표준화)

$$
x^\prime
\;=\;
\frac{x - \mu}{\sigma}
$$

- Transforms data to have mean 0 and standard deviation 1.
- Often used when data is assumed to be approximately normal.

---

## 5. t-test and ANOVA

### Null & Alternative Hypotheses

- **Null (H₀)**: baseline assumption, e.g. “means are the same”.
- **Alternative (H₁)**: e.g. “means differ”.
- If \(p<0.05\), reject \(H₀\).

### t-test

- Compares **two** group means.
- **Paired**: same subjects before/after.
- **Independent**: two separate groups.

### ANOVA

- Compares **3 or more** group means.
- One-way, two-way, multi-factor, etc.
- If significant, do **post hoc** tests to compare specific pairs.

**Key conditions**: normality, homogeneity of variance, independence.

---

## 6. Normal Distribution (정규 분포)

A continuous distribution \(N(\mu, \sigma^2)\):

$$
f(x)
\;=\;
\frac{1}{\,\sqrt{\,2\pi\,}\,\sigma}\,
\exp\!\Bigl(-\tfrac{(x - \mu)^2}{2\,\sigma^2}\Bigr).
$$

### Standard Normal (표준 정규 분포)

$$
Z
\;=\;
\frac{X - \mu}{\sigma}
$$

### Multivariate Gaussian

- Extends to multi-dimensional \(\boldsymbol{\mu}\) and \(\Sigma\).

---

## 7. Bayes' Theorem (베이즈 정리)

$$
P(H\mid E)
\;=\;
\frac{P(E\mid H)\,P(H)}{P(E)}
$$

- Updates prior \(P(H)\) to posterior \(P(H \mid E)\) given evidence \(E\).

---

## 8. Maximum Likelihood Estimation (MLE)

- Finds parameters (e.g., \(\mu\), \(\sigma\)) maximizing the **likelihood** of observed data.
- Usually maximize log-likelihood for computational convenience.

---

## 9. Principal Component Analysis (PCA)

- Finds new axes (principal components) with maximal variance.
1. Center data (subtract mean).
2. (Optionally) scale by std dev.
3. Compute covariance matrix \(\Sigma\).
4. Decompose \(\Sigma\) → eigenvectors (principal axes) & eigenvalues (variances).

Used for **dimensionality reduction** and decorrelation of features.

---

## 10. Regression Analysis

### Linear Regression

$$
\hat{y} = wx + b
$$

- Minimize squared error \(\sum (y - \hat{y})^2\).
- Analytical solutions or gradient-based methods exist.

### Logistic Regression

- For **binary classification**.
- Uses **sigmoid** to map \((-\infty, +\infty) \to (0, 1)\).
- Loss: binary cross-entropy.

### Softmax Regression

- **Multi-class** extension of logistic regression.
- Softmax function normalizes logits into probabilities.
- Loss: cross-entropy for one-hot labels.

### Polynomial Regression

- Uses polynomial terms \(x^2, x^3, \dots\) to model nonlinear data.
- Interpolation methods include Lagrange, Newton, spline, etc.

---

## 11. Perceptron (퍼셉트론)

- A linear model that outputs 0 or 1 based on a weighted sum vs. threshold:
  
$$
\text{output} = \begin{cases} 
      1, & \mathbf{w} \cdot \mathbf{x} > b \\
      0, & \mathbf{w} \cdot \mathbf{x} \le b
   \end{cases}
$$
- **XOR** is not linearly separable → leads to multi-layer networks.

---

## 12. Activation Functions

### Step Function

$$
\text{step}(x) \;=\;
\begin{cases}
1, & x \ge 0\\
0, & x < 0
\end{cases}
$$

- Not differentiable at 0, derivative is 0 otherwise → no gradient flow.

### Sigmoid

$$
\sigma(x) \;=\; \frac{1}{\,1+e^{-x}\,}
$$

- Smooth curve from (0,1).
- Suffers from vanishing gradients in deep nets.

### Tanh

$$
\tanh(x) \;=\; \frac{\,e^x - e^{-x}\,}{\,e^x + e^{-x}\,}
$$

- Range \((-1,1)\).
- Zero-centered but still can vanish in deep layers.

### ReLU

$$
\mathrm{ReLU}(x) \;=\; \max(0, x)
$$

- Simple, positive part has gradient = 1.
- **Dying ReLU**: negative inputs yield 0 → gradient stuck.

### Leaky ReLU

$$
\mathrm{LeakyReLU}(x)
\;=\;
\begin{cases}
x, & x \ge 0\\
\alpha x, & x < 0
\end{cases}
$$

- Lets a small slope \(\alpha\) for \(x<0\).

### PReLU, ELU, GELU

- **PReLU**: slope \(\alpha\) is learnable.
- **ELU**: exponential decay for negative side, reduces bias shift.
- **GELU**: used in BERT/GPT, approximates Gaussian error function.

---

## 13. Gradient Descent & Backpropagation

### Gradient Descent

$$
w
\;\leftarrow\;
w - \alpha\;\frac{\partial E}{\partial w}
$$

- Moves \(w\) in direction that reduces loss \(E\).
- \(\alpha\) is the learning rate.

### Backpropagation

- Applies the **chain rule** to compute \(\frac{\partial E}{\partial w}\) from the output layer backward.
- Key to training deep neural networks.



# korean.ver
# Mathematics-used-in-AI
# 평균, 기댓값, 분산
## 평균(mean)
 평균은 자료를 대표하는 값으로 모든 데이터 값을 더한한 것을 데이터의 수로 나눈것 평균의 종류로는 산술 평균, 조화 평균 기하 평균이 있다. 조화 평균은 기준이 다를때 기준을 통일 시키기 위해 역수를 평균낸다음 다시 역수를 취하주는 것이다. 기하 평균은 예를 들어 종이를 1/2복사하고 2배를 복사했다치다 그럼 이종이는 평균 몇 배 복사 되었다? -->루트(1/2*2)=1 즉, n개의 양수 값을 모두 곱한 값을 n제곱근한 값이다.

## 기댓값
 확률적 사건에 대한 평균값이다. 전체 사건에 대해 사건이 벌어졌을 때의 값과 그 사건이 벌어질 확률을 곱한 총값  
평균과 기댓값은 같다고 볼 수 있다. 굳이 차이를 두자면 기댓값은 확률적으로 접근한 값이다. 따라서 기댓값은 시행을 무한히 반복하면 평균이된다.  
ex)Minst 데이터셋에서는 평균이 아닌 기댓값이라고 표기한다. 왜 이렇게 표기하는가?-> 7만장의 데이터중 하나를 뽑았을 때 어떤 수가 뽑힐 확률 즉 확률적으로 접근하였기 때문 즉, 7만장의 샘플이 모집단의 평균을 대변한다.(확률적으로 접근한다. 7만장의 시행의 기댓값....)

## 분산
데이터가 평균값에서부터 퍼져있는 평균 거리  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/0fe592a1-0bc9-4ff2-9780-4c58cc6f7151)(만약 표본 분산이라면 n이아니라 n-1이다.)  
평균과 분산은 계산하기에 용이하고 통계적 결과를 추정하는데 정말 용이 하기 때문에 많이 쓰인다.(ex 정규분포)

# MSE, RMSE, MAE
## MSE
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/f9afa197-4ec5-4bba-89cb-0bf6f917efc3)  
주로 회귀 모델에서 사용하는 손실함수  
이차 함수 형태로 모든 함수값에서 미분이 가능하다. 이상치에 대해 민감하며 잘 다룬다. 즉 모델을 학습 시킬 때 사용한다.  
MSE는 제곱 형태이기 때문에 오차가 크면 클수록 값은 그것보다 더 커진다. 즉 왜곡에 민감하다.

## RMSE
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/138b3439-b98e-4ba3-ab24-3334e513eeb5)  
마찬가지로 회귀 모델에서 사용하는 함수 이다.
MSE에서 제곱에 의해 생기는 문제점들을 어느정도 해소 시킨다. 이상치에 대해 적절히 잘 다룬다. 그러나 모델을 학습 시킬 때는 적절하지 않다. 따라서 모델을 평가할 때 사용한다.

## MAE
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/2a6840cd-7893-4616-bcb4-377402bcf919)  
RSME와 마찬가지로 왜곡에 대해 둔감한 함수이다. RMSE보다 더 둔감하다. 따라서 모델을 평가할 때 자주 사용된다.

# 공분산과 상관계수
## 공분산
공분산은 두확률 변수의 선형적 관계를 나타내는 수이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/1ab24e82-8022-472d-9406-ab4f0a3f123e)(이것은 표본 공분산이다. 모 공분산일 때는 n-1이 아닌 n이다.)  
만약 하나의 값이 평균 보다 클때 다른 값도 해당 평균 보다 작거나 크다면 선형관계를 갖는다고 볼 수있다. 또한 한 쪽이 크면 다른 한 쪽도 큰 관계라면 양의 공분산을 가지고 다른 한쪽이 작은 관계라면 음의 공분산을 가진다. 만약 무상관이면 0에 가까운 값을 갖는다. 그리고 범위에도 영향을 받는다.

## 공분산 행렬
만약 확률 변수 n개가 있다면 {x1,x2,...,xn}에 대한 공분산 행렬은 다음과 같이 정의할 수 있다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/7e04941e-f176-4f6e-8eef-8a297b6b7d47)(http://www.ktword.co.kr/test/view/view.php?no=5596)
대각 행렬은 각 확률변수의 분산이고 비대각 행렬은 확률변수들 끼리의 공분산이다.

### 기하학적 의미
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/a93d3807-74c4-49e2-b665-231585011295)  
x=[1,3,3,5] y=[1,3,1,3]이고 공분산은  
[2,1]  
[1,1]  
이다. 이 공분산 행렬의 고유 벡터와 고윳값은 위 사진에 적혀있는것과 같다. 즉 고유벡터의 방향 벡터는 데이터들의 분산의 방향이고 고윳값은 분산의 크기이다. 또한 원형으로 분포되어 있는데이터를 공분산 행렬로 선형변환 시켜 저렇게 변한다고도 볼 수 있다.

## 상관계수
상관계수는 상관관계의 정도를 수치화한 수로 다음과 같이 나타낼 수 있다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/ae7e1369-89ea-4cbe-956e-e72cd495c0ed)

#  normalization,standardization
모델을 학습 전에 scaling하는 것 scale이 큰 feature에 의해 그 영향이 비대해지는 것을 방지해준다.
## normalization(정규화)
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/a9af0a56-fab1-444f-8950-9f019662841c)  
데이터셋의 범위차이를 왜곡하지 않고 공통 척도로 변경한다.  
최대값 및 최소값을 이용하ㅣ여 스케일링한다. feature의 범위가 다를 때 사용한다. 주로 [0~1]로 스케일링한다. 상대적 크기에 대한 영향력을 줄여준다.

## standardization(표준화)
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/ee5403ef-c767-4a95-959e-156a5c39ee94)  
평균이 0이고 표준편차가 1인 표준 정규 분포의 속성을 갖도록 변경한다.  
평균과 표준편차로 스케일링 한다. 데이터 분포를 정규분포 형태로 변환시 사용한다. 특정범위에 딱히 제한되어 있지 않다. 이상치를 제거할 때도 사용된다.

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

## 1.Linear Regression(선형 회귀)
인공지능에서 가장 일반적으로 쓰이는 회귀 분석 선형 함수를 사용하여 독립변수와 예측변수간의 관계를 도출해냄 행렬로 나타내면 아래와 같다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/401d1ec7-6761-4e64-a8a8-46e845f9b6b5)  
또한 회귀식이 p=wx+b일때 실제 값하고의 차이는 다음과 같다. error={(y-(wx+b)}^2 error가 최소가 되는 w와 b를 구하는 방법은 다음과 같다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/8056b4a7-e00c-4398-9b02-e0f9141a4359)  
위의 만들어진 1번과 2번식을 연립하면 w와 b를 구할 수 있다.

## 2.Logistic regression
종속 변수가 이진형일 때(1과 0) 사용하는 회귀이다. 선형회귀에서는 범주형 자료가 이진형이면 관계를 추정하기 어렵다.  
Logistic regression은 시그모이드 함수라는 것을 이용한다. 이 시그모이드 함수는 -무한에서 +무한까지의 값을 0과 1사이의 값으로 바꿔준다.

### probability,odds,logit
odds는 실패확률에 대한 성공 확률의 비율이다. 수식으로 나타내면 p(success)/(1-p(success))이다.  
이 odds에다 로그를 취하면 log(p(success)/(1-p(success)))가 된다.  
p[1~0]->odds[0~무한]->logit[-무한~+무한]으로 저과정은 1과0 사이의 확률을 -무한에서 +무한으로 범위를 바꿔주는 과정이다. 이 과정을 회귀식에 적용하여 거꾸로 하면 어떻게 될까? 다음은 역함수를 취해 주어 이과정을 거꾸로 하는 과정이다.  

![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/6d342ea8-b75c-4d5c-8051-cb06947c7a57)  
최종적으로 저런 식이 나오게되는데 저것이 바로 [-무한~+무한]의 값을 [0~1]의 값으로 바꿔주는 시그모이드 함수이다. "step function(계단 함수)를 쓰면 되지 저런 형식의 식이 필요할까?" 라고 생각할 수 있다. 그러나 step function 제대로 된 기울기가 형성되있지 않아(기울기가 0이다) 모델의 오차를 학습하는데 어려움이 있다.

### Loss function
Logistic regression은 손실 함수로 BCE(Binary cross entropy)라는 함수를 쓴다. 형태는 이러하다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/187b3f75-2f52-4dc5-820b-6d45ea4b9712)  
y=정답값,y헷=예측값이다. 예측값이 정답에 가까울 수록 0에 근접하고 예측값이 정답에서 멀어질수록 +무한으로 발산한다.

## 3.Softmax regression
Softmax regression는 종속변수가 이진형이 아닌 그것 보다 많은 범주형 데이터에서 사용하는 회귀이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/0bb33a4a-dd45-4914-8969-2a7fff98fa9e)  
Softmax regression은 위의 사진과 같은 함수를 쓴다. 이 함수는 [-무한~+무한]의 값을 지수를 이용해 [0~+무한]의 값으로 바꿔준 다음에 [0~1]의 범위로 정규화 해준 것이다.

### Loss function
Softmax regression는 one-hot encoded cross entropy라는 것을 사용한다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/deee7cf3-7d1e-483d-8144-49659ef7e312)

## 4.Polynomial regression
선형 모델을 사용하여 비선형 데이터 집합을 모델링 하는것 이 글에서는 interpolation(보간법)위주로 설명한다.

### interpolation(보간법)
#### polynomial interpolation
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/9fcdffd9-17f8-4157-b619-89f056a3245b)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/fb688436-d275-44b2-8c65-dbc8cabd1903)  
이 방법은 모델을 구하는 기본적인 방법이다. 그러나 계산이 너무 복잡하다는 문제가 있다.

#### Newton's interpolation
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/3e7de992-339f-45ad-bfec-8a5cc2dcda8a)  
이것은 뉴턴 보간법이다. 위의 방법보다 계산이 쉽다. 또한 예시로 든 3개의 데이터에서 새로운 데이터가 추가되어도 새로운 데이터에 관한것만 계산하면 되기에 시시각각 변하는 능동적인 모델에 적합하다.

#### Lagrangian interpolation
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/747e9815-3bc3-4b0b-a92d-f03004249998)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/9f54ece0-04d4-4818-b9a2-ca798769912a)  
이것은 라그랑지에 보간법이다. 위의 방법과 다르게 선형 시스템이 필요없다는 장점이 있어서 계산이 더 간편하다. 하지만 새로운 데이터가 추가되면 처음부터 다시 계산해야되기 때문에 시시각각 변하는 능동적인 모델에 적합하지 않다.

### Spline interpolation
Spline interpolation은 점과 점사이마다 함수가 있는 것이다. Spline interpolation은 대표적으로 Linear spline, Quadatic spline, Cubic spline있다.

#### Linear spline
Linear spline는 점과 점사이의 함수가 1차 함수인 것으로 아래 사진 처럼 점과 점사이의 직선을 이용하여 구한다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/acf460f9-459b-4674-964e-54f4679ae115)

#### Quadratic spline
Quadratic spline는 점과 점사이의 함수가 2차 함수인 것으로 아래의 식을 이용해 연립방정식을 만들어 구한다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/b1d6155f-11eb-42cd-b8fe-1824eae51fb7)  
위의 사진에서 함수가 3개이므로 3*n(n=3)개의 미지수가 있으므로 총 3n개의 방정식이 필요하다. 위의 식에서 처럼 점을 함수식에 대입했을 때에 2n개의 수식이 나온다. 이루 끝점을 제외한 점에 이어져있는 두함수의 기울기를 같게하여 n-1개의 수식을 얻어준다. 이후 1개의 수식이 부족하므로 양쪽의 함수중 하나를 직선의 방정식(a=0으로 만듬)으로 만들어 나머지 하나의 수식(a1=0 or a3=0)을 얻어 총 3n개의 수식을 얻어준다. 이렇게 얻은 수식을 이용하여 함수식을 구할 수 있다.

#### Cubic spline
Cubic spline는 점과 점사이의 함수가 3차 함수인 것으로 아래의 식을 이용해 연립방정식을 만들어 구한다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/1159fb2b-0809-44fd-87bc-4606cd12c80d)  
cubic의 함수식을 구하는 것은 Quadratic spline과 비슷하다. 그러나  Quadratic spline이 양쪽의 함수를 두번 미분한것중 하나를 0으로 만든것과 다르게 양쪽의 함수를 두번 미분한것을 모두 0으로 두어 수식을 얻는다.

# perceptron(퍼셉트론)
여러 입력값을 받아 가중치를 곱하고 그 결과를 합한 값이 임계치를 넘으면 1또는 0을 출력하는 선형 시스템  
ex) x1*w1+x2*w2 <=b-->b보다 작으면 0 크면 1 반환

## AND gate
2개의 입력 x1,x2가 모두 1일 때 1을 반환  
ex)(1,1)->1 (1,0)->0  
x1*0.4+x2*0.4<=0.7: return 0 x1*0.4+x2*0.4>0.7: return 1

## OR gate
2개의 입력 x1,x2 둘 중 하나라도 1이면 1반환  
ex)(1,1)->1,(1,0)->1.(0,0)->0  
x1*0.4+x2*0.4<=0.3: return 0 x1*0.4+x2*0.4>0.3: return 1

## NAND gate
2개의 입력 x1,x2가 모두 1일 때 0반환 나머지는 1반환  
ex)(1,1)->0 (1,0)->1
x1*(-0.4)+x2*(-0.4)<=0.6: return 0 x1*(-0.4_+x2*(-0.4)>0.6: return 1

## XOR gate
2개의 입력 x1,x2 (1,0),(0,1)일때만 1반환  
하지만 이것은 기존 perceptron으로 구현할 수 없음 단 다중 퍼셉트론으로 가능  
NAND를 여러개 구성해 구현가능, NAND,OR,AND를 적절히 조합하여도 구현가능->선형 시스템 perceptron을 조합하여 XOR해결->deep learning 원형

# Activation Function(활성화 함수) 
Activation Function은 비선형성을 추가해주어 Hidden layer(은닉층)에 의미를 부여해준다. 만약 Activation function이 존재하지 않는다면 층을 쌓는 의미가 없어진다. 즉 층을 아무리 많이 쌓아도 층이없는 신경망이랑 다를바가 없다는 것

## step function(계단 함수)
perceptron 에서 사용하는 activation function이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/32a8e1da-691a-4b22-91ec-efe988e8d690)  

### boundary problem
step function은 0.000000001과 -0.000000001은 다르게 보고 0.00000001과 10은 같게 본다.

### 미분 문제
step function은 임계치 부분에서 불연속이기 때문에 미분이 불가능하다. 이 부분만 제외하고는 미분이 가능하지만 제일 큰 문제는 모든 부분을 미분해도  미분값이 0이라는 것이다. 즉 가중치를 업데이트 하지 못한다.

## sigmoid function
step function 과 유사함, 연속적이어서 모든 부분에서 미분이 가능하고 step function과 다르게 미분값이 0이 아님, 미분 결과가 간편함 f'=(1-f)f f:sigmoid, 0~1의 값을 가지기 때문에 확률 개념과 결합 가능  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/72451f3a-a986-4846-a94d-4189b92a90a3)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/e505168c-2ae0-4646-b90d-feccd63d0279)  

### zigzag problem
우선 zigzag와 Gradient vanishing을 설명하기 위해 가중치의 업데이트하는 과정이 어떻게 이루어지는지 확인하겠다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/f8fce4c3-b14c-4f9e-8acd-b5e23c41f2f9)  
이런식으로 미분을 통해 가중치가 업데이트 된다. 위의 수식을 보면 가중치가 업데이트 되는 방향이 -(y1-a21)(손실함수를 미분한값)에의해 결정되는 것을 볼 수 있다.( f'()시그모이드의 미분값이므로 항상 0보다 크다. a11은 sigmoid를 통과한 값으로 이 역시 항상 0보다 크다.) 따라서 가중치들이 업데이트될 때 모두 같은 방향으로 업데이트되기 때문에 최적의 w를 비효율적으로 찾는다.

### Gradient vanishing  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/fa395b45-e1b4-46bf-8484-2d7ce788a14d)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/32ef0975-a412-4f30-a8ca-e21e99ff67de)  
신경망에서 층을 거듭할 수록 오차를 전달하지 못하는 문제이다. 위식을 보면 층이 깊어질수록 sigmoid를 미분한 값을 계속 곱해주는 것을 볼 수 있다. sigmoid의 미분값은 항상 1보다 작기 때문에 곱해지면 곱해질수록 0으로 수렴한다. 즉 오차가 제대로 전달이 안된다.

## hyperbolic tangent(tanh)
sigmoid와 유사함  
sigmoid보다 학습속도가 빠름 그 이유는 tanh가 zero centered되 있기 때문 즉, 0이 나오는 값은 베제되기 때문에  
sigmoid보다 이론상 층을 4배 깊게 쌓을 수 있지만 여전히 gradient vanishing 발생0  
sigmoid와 다르게 양수와 음수 모두 출력할 수 있어 zigzag problem 해결
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/8bc410fc-fb06-4322-a229-f7b996322490)

## rectified linear unit(ReLU)
미분 했을 때 0보다 크면 항상 1이 나오기 때문에 Gradient vanishing 해결  
간단한 함수의 형태와, 0보다 작은 값들은 모두 버리기 때문에 학습속도가 아주 빠름  
신경망을 깊게 쌓을 수 있음  
단,sigmoid처럼 zigzag problem 발생  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/0eb4bc10-e7a9-47bf-94d7-2487b928d0f1)  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/dccede1b-49c4-4eb5-9029-743a47202518)

### dying ReLU
0보다 작은 값들은 모두 0이되어 정보가 소실되는 문제이다. 즉, 0보다 작은값들이 많이 나오게 되면 파라미터의 업데이트가 원할하지 않다.

### Leaky ReLU
0보다 작은 값들에게 조금의 기울기를 주어 dying RELU문제를 해결한 함수이다.
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/923464d6-4259-485a-a290-55687d86b423)

### parametric ReLU(PReLU)
각 레이어마다 활성화 함수의 a값을 학습시키는 함수이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/6109e648-250a-4dca-a11b-29e10a81c13f)

### exponential linear unit(ELU)
너무 작은 음수들은 학습에 잘 참여 시키지 않게 고안된 함수이다. 성능이 매우 우수하다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/00741789-5ddf-47a1-86a8-8d7b1a03790f)  
->https://pytorch.org/docs/stable/generated/torch.nn.ELU.html

### gaussian error linear unit
BERT나 GPT에 사용되는 Activation function으로 정규분포를 근사화 하는 함수. 메우 깊은 신경망에서 ReLU보다 높은 정확도를 보인다.
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/5a83b683-9f1b-4f1a-9fcb-4177e3305f4d)  
->https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

# Gradient Descent(GD,경사하강법) && Backpropagation
경사하강법은 손실함수의 값이 낮아지는 방향으로 가중치를 변환시켜 나아가는 방법이다. 가중치를 업데이트 다음과 같은 수식을 이용한다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/3bc6701a-74cf-4fee-a1fb-ac106a959ca3)  
w0는 이전 가중치 값 a는 학습률, E 정답값과 예측값의 손실함수다.

## 손실함수
신경망에서 계산된 값과 실제 나와야하는 값의 차이를 계산하는 함수 파라미터를 업데이트할 때 사용된다.  
MSE,KL-divergence,JSD,BCE등이 있다. 이런 손실함수에서 나온 오차값을 이용해 파라미터를 업데이트 한다.

## chain rule(연쇄법칙)
y=f(x)=x^2+3이고  
t=x^2이라하자. 또한 y는 t에대해 미분 가능하고 t는 x에대해 미분 가능하다.  
이것을 미분할 때 다음과 같이 쓸 수 있다.  
dy/dx=dy/dt*dt/dx  
이것이 chain rule이다.

다음은 오차를 이용해 가중치를 업데이트하는 과정이다.  
![image](https://github.com/chanyoung-shin/Mathematics-used-in-AI/assets/165111440/f8fce4c3-b14c-4f9e-8acd-b5e23c41f2f9)
