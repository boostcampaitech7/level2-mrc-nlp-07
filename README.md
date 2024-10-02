# Open-Domain Question Answering(ODQA)

## 📕프로젝트 개요

* 부스트캠프 AI Tech `NLP`분야에서 개체된 level 2 대회
* `Linking MRC and Retrieval: Open-domain Question Answering(ODQA)` Task.
  * ODQA: 지문이 따로 주어지지 않은 채로 방대한 World Knowledge에 기반하여 질의응답
  * 질문에 관련된 문서를 찾는 `retriever`와 문서를 읽고 적절한 답변을 찾거나 만드는 `reader`의 `two-stage`로 구성.
* 학습 데이터셋은 3,952개, 검증 데이터는 240개, 테스트 데이터는 600개로 테스트 데이터 중 40%만 Public으로 반영 대회 종료 후 Private 점수가 공개됨.
  * `id` : 질문의 고유 id
  * `question` : 질문
  * `answers` : 답변에 대한 정보, 하나의 질문에 하나의 답변만 존재.
    * `answer_start` : 답변의 시작 위치
    * `text` : 답변의 텍스트
  * `context` : 답변이 포함된 문서
  * `title` : 문서의 제목
  * `document_id` : 문서의 고유 id

* `Exact Match`, `F1 Score`를 통한 평가.

## 📆세부일정

* 프로젝트 기간(4주) : 09.30(월) ~ 10.24(목)

## 😁팀소개

<table style="width: 100%; text-align: center;">
  <tr>
    <th>강감찬</th>
    <th>이채호</th>
    <th>오승범</th>
    <th>이서현</th>
    <th>유채은</th>
    <th>서재덕</th>
  </tr>
  <tr>
    <td><img src="./docs/image/README/꼬부기1.png" alt="꼬부기1" width="100" height="100"></td>
    <td><img src="./docs/image/README/꼬부기2.png" alt="꼬부기2" width="100" height="100"></td>
    <td><img src="./docs/image/README/꼬부기3.png" alt="꼬부기3" width="100" height="100"></td>
    <td><img src="./docs/image/README/꼬부기4.png" alt="꼬부기4" width="100" height="100"></td>
    <td><img src="./docs/image/README/꼬부기5.png" alt="꼬부기5" width="100" height="100"></td>
    <td><img src="./docs/image/README/꼬부기6.png" alt="꼬부기6" width="100" height="100"></td>
  </tr>
  <tr>
    <td><a href="https://github.com/gsgh3016">@감찬</a></td>
    <td><a href="https://github.com/chell9999">@채호</a></td>
    <td><a href="https://github.com/Sbeom12">@승범</a></td>
    <td><a href="https://github.com/seohyeon0677">@서현</a></td>
    <td><a href="https://github.com/canolayoo78">@채은</a></td>
    <td><a href="https://github.com/jduck301">@재덕</a></td>
  </tr>
  <tr>
    <td>고독한 음악인 감찬</td>
    <td>혼자있는 지방러 채호</td>
    <td>살을 빼야하는 승범</td>
    <td>귀염둥이 막내 서현</td>
    <td>야구를 볼 수 있는 채은</td>
    <td>오리는 꽥꽥 재덕</td>
  </tr>
</table>

## 프로젝트 수행 절차 및 방법

## 프로젝트 아키텍쳐


## 프로젝트 결과

## Getting Started

## Appendix

||Ensemble 전|Ensemble 후|
|:-:|:-:|:-:|
|pearson 점수|0.9175|0.9360|

### 프로젝트 폴더 구조

### 협업방식

* Notion
* Git
* Jira
