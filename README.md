# Instead_of 프로젝트

실시간 speech-to-text 를 통해 강의,연설,대화 내용을 요약/정리 하고, 
RAG 를 통해서 해당 내용들을 이해하기 더 쉽게 만들어 줄 뿐만 아니라 사용자와 질의응답을 통해 이해를 풍부하게 도와준다. 

순서:
crop -> img_text_table -> relocation 

1. RAG

1단계 : parser - doc layout yolo 사용 -> 텍스트,이미지,표 매우 상세하게 나눌 수 있을 뿐만 아니라 비용이 저렴 ( 텍스트 -> figure , just text , abandon 등등..... )


2. speech-to-text
모델후보
1) openai whisper : 성능 좋음 / api 비용 내야함
2) 




Add more
1. stt 를 더 빠르고 정확하게 처리하는 능력
2. RAG 의 성능높이기
3. 추가적인 기능 : 컴퓨터 내 혹은 모든 사운드가 아니라 원하는 사운드만 녹화할수 있도록 ( 컴퓨터를 켜두고 다른 행동을 할 수 있도록 )
4. 

fix more 
![image](https://github.com/user-attachments/assets/251ff454-6255-4fa2-906c-e3b81cf70a05)
다음과 같은 경우를 이미지라고 판단하는 경향이 있음 .이미지보단 표에 가깝지 않나 싶기도 한데 
doc yolo 에서는 정통 표만 튀급하는 것 같긴함 
그래프를 어떻게 해석할지도 고민. 

< test pdf > : 유튜브 보다가 이게 복잡한 pdf 라서 사용한다는 걸 보고 나도 이거로 사용하기로 함.
추가로 여러가지 문서로 실험 예정.

[sample.pdf](https://github.com/user-attachments/files/18793775/sample.pdf)
