from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import PyMuPDFLoader 

from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders import PDFMinerLoader 


pdf_path = r"C:\Users\xpc\Desktop\CLIP.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()


# 로드된 문서는 Document 객체 리스트로 반환됨
print(f"총 {len(documents)}개의 문서 청크가 로드되었습니다.")
print("첫 번째 청크의 내용 일부:")
print(documents)  # 첫 500자 출력