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

# 정규 분포
