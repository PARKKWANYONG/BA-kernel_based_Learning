# BA-kernel_based_Learning
Support Vector Machines Classifier Tutorial with Python 
Support Vector Machines(SVM)은 분류 및 회귀 목적으로 사용되는 지도학습 머신러닝 알고리즘입니다.
저는 Iris Dataset과 Iris Dataset에 비해 상대적으로 복잡한 Breast_Cancer Dataset에서 SVM을 활용하여 분류문제를 해결하고자 했습니다.
 
# 1. Introduction to Support Vector Machines
 
 Support Vector Machine(SVM)은 분류 및 회귀 목적으로 사용되는 기계 학습 알고리즘입니다. SVM은 분류, 회귀 및 이상값 감지를 위한 알고리즘입니다. SVM은 구축된 마진에서 새 데이터 포인트를 할당하는 모델을 구축합니다. SVM 알고리즘은 Vladimir N Vapnik과 Alexey Ya가 개발했습니다. 1992년 Bernhard E. Boser, Isabelle M Guyon 및 Vladimir N Vapnik은 커널 트릭을 최대 마진 초평면에 적용하여 비선형 분류기를 만드는 방법을 제안했습니다. 현재 사용되고 있는 표준 SVM은 1993년 Corinna Cortes와 Vapnik에 의해 제안되었고 1995년에 Summit되었습니다. SVM은 선형 분류를 수행하는 것 외에도 커널 트릭을 사용하여 비선형 분류를 효율적으로 수행할 수 있습니다. 이를 통해 입력을 고차원 공간에 매핑할 수 있습니다.
 
# 2. Support Vector Machines intuition
 
 초평면(Hyperplane) 
 
초평면은 클래스 레이블이 서로 다른 주어진 데이터 포인트 집합 사이를 구분하는 결정 경계입니다. SVM 분류기는 최대 여백이 있는 초평면을 사용하여 데이터 포인트를 분리합니다. 이 초평면을 최대 마진 초평면이라고 하며 정의하는 선형 분류기를 최대 마진 분류기라고 합니다. 
 
 서포트 벡터 (Support Vectors)
 
서포트 벡터는 초평면에 가장 가까운 샘플 데이터 포인트입니다. 이러한 데이터 포인트는 여백을 계산하여 분리선 또는 초평면을 더 잘 정의합니다. 
 
 마진 (Margin)
 
마진은 가장 가까운 데이터 포인트에서 두 줄 사이의 분리 간격입니다. 벡터 또는 가장 가까운 데이터 포인트를 지원하는 선에서 수직 거리로 계산됩니다. SVM에서는 이 분리 간격을 최대화하여 최대 마진을 얻으려고 합니다. 아래 그림을 통해 더 직관적인 이해가 가능할 것 입니다.

![image](https://user-images.githubusercontent.com/71392868/199665408-2e5979ec-832d-491c-a96a-617eabf4da94.png)

# 3. Kernel trick

비선형 SVM 알고리즘은 커널트릭을 사용하여 구현됩니다. 커널은 데이터를 분리할 수 있는 더 높은 차원으로 데이터를 매핑하는 기능입니다. 커널을 통해 저차원 입력 데이터 공간을 고차원 공간으로 변환합니다. 따라서 더 많은 차원을 추가하여 비선형 분리 가능한 문제를 선형 분리 가능한 문제로 변환합니다. 따라서 커널 트릭은 정확한 분류기를 만드는 데 도움을 줍니다.

![image](https://user-images.githubusercontent.com/71392868/199665787-89abeecf-ef77-454b-aa82-1aebb5adff9d.png)

# 3.1 Linear kernel

linear kernel : K(xi , xj ) = xiT xj

선형 커널은 데이터가 선형으로 분리될 때 사용됩니다. 가장 일반적으로 사용되는 커널 중 하나입니다. 
선형 커널은 아래 그림으로 더 직관적으로 이해할 수 있습니다. 

![image](https://user-images.githubusercontent.com/71392868/199666167-2e345a54-f85e-490c-857d-105e4830e340.png)

# 3.2 Polynomial Kernel

For degree-d polynomials, the polynomial kernel is defined as follows –

Polynomial kernel : K(xi , xj ) = (γxiT xj + r)d , γ > 0

Polynomial 커널은 자연어 처리에서 자주 사용됩니다. 
차수가 클수록 자연어 처리 문제에 과적합되는 경향이 있기 때문에 차수는 보통 2로 설정합니다. 
Polynomial 커널은 아래 그림으로 더 직관적으로 이해할 수 있습니다.

![image](https://user-images.githubusercontent.com/71392868/199666592-44488640-56ce-4e6d-aa4a-b63a067746dd.png)

# 3.3 Radial Basis Function Kernel

RBF 커널은 데이터에 대한 사전 지식이 없을 때 사용됩니다.
두 개의 샘플 x 및 y에 대한 RBF 커널은 다음 방정식으로 정의됩니다.

![image](https://user-images.githubusercontent.com/71392868/199666753-c78d7616-2987-4400-b2d0-6c1bfa49a7d7.png)

RBF 커널은 아래 그림으로 더 직관적으로 이해할 수 있습니다.

![image](https://user-images.githubusercontent.com/71392868/199667638-c8a6ec67-b525-41cd-9f5b-3e23c4cd6f22.png)



# 3.4 sigmoid kernel

시그모이드 커널은 뉴럴 네트워크에서 차용되었습니다.
시그모이드 커널은 아래와 강큰 방정식으로 정의됩니다.

sigmoid kernel : k (x, y) = tanh(αxTy + c)

sigmoid 커널은 아래 그림으로 더 직관적으로 이해할 수 있습니다.

![image](https://user-images.githubusercontent.com/71392868/199667505-bfd63e03-f3af-478c-b5c9-fcba597ff303.png)




Reference1 : https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook#Support-Vector-Machines-Classifier-Tutorial-with-Python



Reference2 : https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/



Reference3 : https://datascienceschool.net/03%20machine%20learning/09.01%20%EB%B6%84%EB%A5%98%EC%9A%A9%20%EC%98%88%EC%A0%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0.html


