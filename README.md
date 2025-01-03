# BERT Tabanlı Soru-Cevap Sistemi

Bu proje, **Mersin Üniversitesi** öğrencilerinin Uygulamalı Eğitimler ile alakalı sorularını cevaplamak amacıyla geliştirilmiştir. Bir transformer modeli olan **BERT** kullanılarak geliştirilmiş bir **Soru-Cevap Sistemi**'dir. Kullanıcıların Türkçe sorularına anlamlı cevaplar sunmayı hedefler. Uygulama, **Streamlit** arayüzü ile entegre edilmiştir ve **BERT** modelini ve **Sentence-BERT**'i kullanarak doğal dil işleme (NLP) ve derin öğrenme yöntemlerini kullanarak soruları analiz eder ve uygun cevapları üretir.

## Proje Hakkında

Bu uygulama, kullanıcan soruları alarak, **Sentence-BERT** ile en uygun cevabı arar ve fine-tuned edilmiş **BERT** modeli ile soruya uygun cevabı döndürür. Sentence-BERT, kullanıcı tarafından sağlanan soruya uygun bağlamları, önceden tanımlanmış bir veri kümesinden (Excel dosyası) bulur. Ayrıca BERT modeli bu veri kümesi ile eğitilmiştir.

### Kullanılan Teknolojiler:
- **BERT** (Bidirectional Encoder Representations from Transformers) modelinin **`BertForQuestionAnswering`** versiyonu kullanılarak soruya uygun cevap döndürülür. Hugging Face kütüphanesinden **[dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)** kullanılmıştır .

- **Sentence-BERT** kullanılarak sorulara uygun bağlamlar benzerlik hesaplaması ile bulunmuştur. Hugging Face kütüphanesinden **[intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)** kullanılmıştır.

- **Streamlit** kullanarak kullanıcı dostu bir web arayüzü oluşturulmuştur.

### Model, Tokenizer ve Excel Dosyası
Bu projede kullanılan fine-tuned edilmiş **BERT** tabanlı model ve tokenizer'ı aşağıdaki Google Drive bağlantılarından indirebilirsiniz:

- **Model**: **[BERT Modeli (Google Drive)](https://drive.google.com/drive/folders/1bX8aHRo9umipMLjKfuh_LRb9aS_jsQhf?usp=sharing)**
- **Tokenizer**: **[Tokenizer (Google Drive)](https://drive.google.com/drive/folders/1-8QdO_7GNxjatLmZG-S0FMHnT-hU7rPe?usp=sharing)**

- Excel dosyasını ise bu **[GitHub Reposundan](https://github.com/obenadak/obenadak-SoruCevaplamaMEU/blob/master/sorucevap.xlsx)** indirebilirsiniz.

> **UYARI:** Model ve tokenizer dosyalarını indirdikten sonra, **`streamlit.py`** dosyanızda bu dosyaların bilgisayarınızdaki yollarını belirtmeniz gerekmektedir.

**Kodda değişiklik yapacağınız satırlar:**
15. satır: 
excel_file_path = "C:/Users/obena/Desktop/sorucevap.xlsx"  # excel dosya yolunuzu girin 

56. satır:
model_path = 'C:/Users/obena/Downloads/model'  # modeli kaydettiğiniz dosya yolunu girin

57. satır:
tokenizer_path = 'C:/Users/obena/Downloads/tokenizer'  # tokenizeri kaydettiğiniz dosya yolunu girin

## Gereksinimler

Bu proje için aşağıdaki Python kütüphaneleri gerekmektedir:

- `streamlit`
- `transformers` (BERT modeli için)
- `torch` (PyTorch, modelin çalışabilmesi için)
- `pandas` (Excel dosyasını okumak için)
- `sentence-transformers` (Sentence-BERT kullanımı için)
- `nltk` (doğal dil işleme için)
- `openpyxl` (Excel dosyalarını okumak için)

> **⚠️ UYARI:** Gerekli kütüphaneleri yüklemek için bash ekranında şu komutu kullanmalısınız:

```bash
**pip install -r requirements.txt**

## Streamlit Uygulamasının Başlatılması

> **⚠️ UYARI:** Streamlit uygulamasını başlatmak için bash ekranında şu komutu kullanmalısınız:

```bash
**streamlit run streamlit.py**

Bu komut, yerel olarak Streamlit uygulaması başlatır ve uygulama web tarayıcınızda çalışmaya başlar.
