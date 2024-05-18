# Manufacturing-Project
공장에서 준 data를 보고 어떤 feature로 비정상으로 여기는지 파악하는 Project
1. csv data에 있는 모든 feature를 plot
2. spindle load, speed data를 하나로 plot해서 비교
3. spindle load, speed의 data를 보니 특정 구간에서 일정한 패턴이 보임(A1: peak 7개, A2: peak 6개)
4. spindle load의 면적과 peak 높이 평균, 분산을 구함
5. IsolationForest로 결과 도출
