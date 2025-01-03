import streamlit as st
import pandas as pd
import torch
import re
import string
from transformers import AutoTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords

if 'stopwords_downloaded' not in st.session_state:
    nltk.download('stopwords')
    st.session_state.stopwords_downloaded = True

excel_file_path = "C:/Users/obena/Desktop/sorucevap.xlsx"  # excel dosya yolunuzu girin 
veri_kumesi = pd.read_excel(excel_file_path)

soru_sutun = 'soru'
cevap_sutun = 'cevap'
tur_sutun = 'tür'

veri_kumesi = veri_kumesi[[soru_sutun, cevap_sutun, tur_sutun]].dropna()
punctions_cluster = string.punctuation
turkish_stopwords = stopwords.words('turkish')
whitelist = {"kim", "mı", "mu", "mü", "nasıl", "ne", "neden", "nerde", "nereye", "nerede",
             "nereye", "niçin", "için", "niye", "her", "de", "da", "ve","biri","en","az","bu"}

def metin_temizleme(metin):
    turkce_cevir = str.maketrans(
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
        "abcçdefgğhıijklmnoöprsştuüvyz"
    )
    metin = metin.translate(turkce_cevir)
    tire_karakterleri = ['-', '–', '—','+', '❝','❛','❜','❞','‘','’',',',"'",'/']
    korunan_karakterler = "()" + "".join(tire_karakterleri)
    metin = re.sub(rf'[^a-zA-Z0-9çÇğĞıİöÖşŞüÜ\s{korunan_karakterler}]', '', metin)

    metin = "".join([
        i if i not in punctions_cluster or i in korunan_karakterler else ""
        for i in metin
    ])

    metin = " ".join([
        kelime for kelime in metin.split()
        if kelime not in turkish_stopwords or kelime in whitelist
    ])

    return metin

veri_kumesi['soru'] = veri_kumesi['soru'].apply(metin_temizleme)
veri_kumesi['cevap'] = veri_kumesi['cevap'].apply(metin_temizleme)
veri_kumesi['tür'] = veri_kumesi['tür'].apply(metin_temizleme)

veri_kumesi['context'] = veri_kumesi['soru'] + " " + veri_kumesi['cevap']

model_path = 'C:/Users/obena/Downloads/model'  # modeli kaydettiğiniz dosya yolunu girin
tokenizer_path = 'C:/Users/obena/Downloads/tokenizer'  # tokenizeri kaydettiğiniz dosya yolunu girin

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

sbert_model = SentenceTransformer("intfloat/multilingual-e5-base")

def soruyu_duzenle(question):
    if "iye" in question.lower():
        if len(question.split()) == 2:
            return question
        if "işyeri eğitimi" not in question.lower():
            question = question.lower().replace("iye", "iye işyeri eğitimi")
    return question

def soru_kelimesi_tespit_et(question):
    soru_kelimeleri = [ 'ne', 'neden', 'nasıl', 'kim', 'hangi', 'mu', 'mü', 'mi', 'mı',
                        'kaç', 'nereye', 'niçin', 'kimin', 'kime', 'kimden', 'kaçıncı',
                        'hangi', 'kimler', 'nerede', 'nereden', 'neresi', 'nereye', 'nereyi',
                        'neyi', 'neyle', 'niye', 'neden', 'neyim', 'neyin', 'neyiz',
                        'nedenim', 'nedenin', 'nedeniz', 'kimim', 'kimin', 'kimiz', 'musun',
                        'müsün', 'hangiyim', 'hanginin', 'hangiyiz', 'müyüm', 'müyüz',
                        'muyum', 'muyuz', 'miyim', 'misin', 'miyiz', 'musunuz', 'müsünüz',
                        'mı', 'mıyım', 'mıysak', 'mısınız', 'misiniz', 'kaçım', 'kaçın',
                        'kaçız', 'nereye', 'nereyim', 'nereyiz', 'niçin', 'kimde', 'kimim',
                        'kiminle', 'hangisi', 'hangisinde', 'hangisiyim',
                        'hangiyle', 'ne zaman', 'neye', 'neler', 'neyle', 'neyleyim',
                        'kendi', 'kendisini', 'kimsenin', 'neci', 'kimce', 'kimlerin',
                        'kimle', 'neyleyebilirim', 'kaçta'
                      ]

    for kelime in soru_kelimeleri:
        if kelime in question.lower():
            return kelime

    return None


def cevap_bul(question, tokenizer, model, veri_kumesi, device, threshold=0.925):
    question = soruyu_duzenle(question)

    if len(question.split()) == 1:
        if question.lower() in ['merhaba', 'selam']:
            return "Merhaba! Size nasıl yardımcı olabilirim?"
        else:
            return "Tek kelimelik sorulara cevap veremem. Lütfen daha açık bir soru sorun."

    soru_kelimesi = soru_kelimesi_tespit_et(question)

    if soru_kelimesi:
        question = metin_temizleme(question)
        questions = veri_kumesi['soru'].tolist()
        contexts = veri_kumesi['context'].tolist()

        question_embedding = sbert_model.encode(question, convert_to_tensor=True)
        question_embeddings = sbert_model.encode(questions, convert_to_tensor=True)

        similarity_scores = util.pytorch_cos_sim(question_embedding, question_embeddings).squeeze()
        best_idx = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_idx].item()

        if best_score < threshold:
            print(f"Benzerlik Skoru: {best_score}")
            return f"Uygun bağlam bulunamadı."

        context = contexts[best_idx]
        print(f"Seçilen Bağlam:\n{context}\nBenzerlik Skoru: {best_score}")
    else:
        return "Bu bir soru değil. Lütfen bana yalnızca soru sorun."

    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()

    if start_idx >= end_idx:
        predicted_answer = "Model cevabı bulamadı"
    else:
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    predicted_answer = predicted_answer.strip()
    print(f"Modelin Cevabı:\n{predicted_answer}\n")
    return predicted_answer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Streamlit Arayüzü
st.set_page_config(page_title="Soru-Cevap Sistemi", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7fa; 
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .title {
            font-size: 50px;
            font-weight: 600;
            color: #FF4500;  
            text-align: center;
            font-family: 'Poppins', sans-serif;
            margin-top: 50px; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2); 
            transition: all 0.3s ease-in-out; 
        }
            
        .subheader {
            font-size: 20px;
            color: #1E90FF;  
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            margin-top: 10px;
            margin-bottom: 30px;
        }

        .emoji {
            font-size: 70px;
            text-align: center;
            margin-bottom: 10px;
        }

        .user-msg {
            background-color: #1E90FF;
            color: white;
            padding: 12px;
            border-radius: 25px;  
            margin: 10px 0;
            max-width: 75%;
            margin-left: auto;
            margin-right: 0;
            display: flex;
            align-items: center;
            position: relative;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1); 
        }

        .bot-msg {
            background-color: #FF6347;  
            color: white;
            padding: 12px;
            border-radius: 25px;  
            margin: 10px 0;
            max-width: 75%;
            margin-left: 0;
            margin-right: auto;
            display: flex;
            align-items: center;
            position: relative;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);
        }

        .user-msg::after {
            content: '';
            position: absolute;
            right: -15px;
            top: 50%;
            transform: translateY(-50%);
            border-width: 10px;
            border-style: solid;
            border-color: transparent transparent transparent #1E90FF;
        }

        .bot-msg::after {
            content: '';
            position: absolute;
            left: -15px;
            top: 50%;
            transform: translateY(-50%);
            border-width: 10px;
            border-style: solid;
            border-color: transparent #FF6347 transparent transparent;
        }

        .msg-icon {
            font-size: 25px;
            margin-right: 10px;
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
        }

        .user-msg .msg-icon {
            right: -40px;  /* Kullanıcı balonunun soluna yerleştirilmiş emoji */
        }

        .bot-msg .msg-icon {
            left: -40px;  /* Bot balonunun sağına yerleştirilmiş emoji */
        }

        .question-input {
            width: 80%;
            padding: 15px;
            font-size: 18px;
            border-radius: 8px;
            border: 2px solid #ccc;
            background-color: #fff;
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            z-index: 999;  /* Keep it above other elements */
            transition: all 0.3s ease-in-out;  /* Smooth transition */
        }

    </style>

    <div class="emoji">🤖</div>

    <div class="title">MEÜ Uygulamalı Eğitimler Soru-Cevap Sistemi</div>

    <div class="subheader">Sorularınızı sormak için aşağıdaki kutuyu kullanabilirsiniz.<br>(Çıkmak için 'çık' yazın.)</div>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

user_question = st.text_input("Soru", "", placeholder="Sorunuzu giriniz... 🔍", key="question", label_visibility="collapsed")

if user_question.lower() == 'çık':
    st.write("Uygulama kapanıyor... :wave:")
    st.write("Yeniden başlatmak için sayfayı yenileyin. :arrows_counterclockwise:")
    st.session_state.messages = []
    st.stop()

if user_question:
    st.session_state.messages.append({"role": "user", "message": user_question})

    answer = cevap_bul(user_question, tokenizer, model, veri_kumesi, device)
    st.session_state.messages.append({"role": "bot", "message": answer})

if not st.session_state.messages:
    st.markdown('''
        <div class="bot-msg">
            <span class="msg-icon">🤖</span>Hoş geldiniz! Sorularınızı sormaktan çekinmeyin. 😊
        </div>
    ''', unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'''
                <div class="user-msg">
                    <span class="msg-icon">👤</span>{msg["message"]}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="bot-msg">
                    <span class="msg-icon">🤖</span>{msg["message"]}
                </div>
            ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
