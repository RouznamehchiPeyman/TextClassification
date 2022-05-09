from tkinter.tix import COLUMN
import streamlit as st

import os
from hazm import *
import itertools
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import glob, os
from pandas_confusion import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import joblib
import pandas as pd

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import AdaBoostClassifier




st.markdown("""
        # Text ***Classification*** Application
        """)

st.write("\n")
st.write("\n")
st.write("\n")

from PIL import Image

photo = Image.open("1.jpeg")
st.image(photo)




# st.header("اپلیکیشن دسته بندی متون")

# st.write("""
#         # اپلیکیشن ***دسته بندی*** متن 
#         """)


st.sidebar.markdown("""
            # Choose Your *ML Classifier*
            """)
st.sidebar.write("""
            
            \n-----------------------\n\n\n
        """)            

clf_svm = st.sidebar.checkbox("SVM")
st.sidebar.write("""
            \n-----------------------\n
        """)

clf_lgb = st.sidebar.checkbox("Lightgbm")
st.sidebar.write("""
            \n-----------------------\n
        """)

clf_mlp = st.sidebar.checkbox("Neural Network")
st.sidebar.write("""
            \n-----------------------\n
        """)

clf_nb = st.sidebar.checkbox("Naive Bayes")
st.sidebar.write("""
            \n-----------------------\n
        """)

clf_lr = st.sidebar.checkbox("Logistic Regression")
st.sidebar.write("""
            \n-----------------------\n
        """)

# st.sidebar.checkbox("Random Forest")
# st.sidebar.write("""
#             \n-----------------------\n
#         """)

# st.sidebar.checkbox("AdaBoost   ")
# st.sidebar.write("""
#             \n-----------------------\n
#         """)        

clf_wve = st.sidebar.radio("Weighted Voting Ensemble", ['None', "Soft Voting", "Hard Voting"])
# st.sidebar.write("""
#             \n-----------------------\n
#         """)

# st.sidebar.selectbox("Boosting", ['AdaBoost', 'Lightgbm', 'None'])
# st.sidebar.write("""
#             \n-----------------------\n
#         """)

# st.sidebar.checkbox("Weighted Voting Ensemble (Soft Voting)")
# st.sidebar.write("""
#             \n-----------------------\n
#         """)

# st.sidebar.checkbox("Weighted Voting Ensemble (Hard Voting)")
# st.sidebar.write("""
#             \n-----------------------\n
#         """)





# st.sidebar.title("SVM")
# st.sidebar.title("Lightgbm")
# st.sidebar.title("Random Forest")
# st.sidebar.title("AdaBoost")
# st.sidebar.title("Neural Network")
# st.sidebar.title("SVM")
# st.sidebar.title("SVM")
# st.sidebar.title("SVM")





def clean(text):
    normalizer = Normalizer()
    # Remove URLs, User Mentions and Hashtags and Retweet and Number
    text = normalizer.normalize(text).replace("\u200c"," ")
    text = ''.join(ch for ch, _ in itertools.groupby(text))
    # Remove Punctuation
#     text = re.sub(r'[^\w\s]', '', text)
    return text.strip()





st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")


st.write("### Please write your sentance")

sen = st.text_input('')
sen = str(sen)
st.write("Your sentance : \n")
st.write(sen)

st.write("\n")
st.write("\n")



def Ptext(text, name_clf, classifier):
    PreText = text
    list2 = []
    text = clean(PreText)
    list2.append(text)
    pred = classifier.predict(list2)
    return (f"{name_clf} : {pred[0]}") 


if len(sen)>10:
    if clf_svm:
        classifier_svm = joblib.load('Svm_Prnw_digi_final.pkl')
        st.write(Ptext(sen, "svm", classifier_svm))
        st.write(40*'-')
    if clf_lgb:
        classifier_lgb = joblib.load('lgb_Prnw_digi_final_GridS.pkl')
        st.write(Ptext(sen, "Lightgbm", classifier_lgb))
        st.write(40*'-')
    if clf_mlp:
        classifier_mlp = joblib.load('MLP_Prnw_digi_final.pkl')
        st.write(Ptext(sen, "Neural Network", classifier_mlp))
        st.write(40*'-')
    if clf_nb:
        classifier_nb = joblib.load('NB_Prnw_digi_final.pkl')
        st.write(Ptext(sen, "Naive Bayse", classifier_nb))
        st.write(40*'-')
    if clf_lr:
        classifier_lr = joblib.load('lgb_Prnw_digi_final_GridS.pkl')
        st.write(Ptext(sen, "Logestic Regression", classifier_lr))
        st.write(40*'-')
    if clf_wve:
        if clf_wve == "Soft Voting":
            classifier_ensemble_S = joblib.load('EnsembleVote_Prnw_digi_final_soft.pkl')
            st.write(Ptext(sen, "Ensemble Vote Classifier(Soft Voting)", classifier_ensemble_S))
            st.write(40*'-')
        elif clf_wve == "Hard Voting":
            classifier_ensemble_H = joblib.load('EnsembleVote_Prnw_digi_final_hard.pkl')
            st.write(Ptext(sen, "Ensemble Vote Classifier(Hard Voting)", classifier_ensemble_H))
            st.write(40*'-')


elif len(sen)<=10 and len(sen)>=1:
    print("ورودی قابل قبول نیست")    




st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
# st.header("Please write your text")
st.write("### Please write your text")
text = st.text_area("")
sen = str(text)
st.write("Your Text : \n")
st.write(text)

st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")



if len(text)>10:
    if clf_svm:
        classifier_svm = joblib.load('Svm_Prnw_digi_final.pkl')
        st.write(Ptext(text, "svm", classifier_svm))
        st.write(40*'-')
    if clf_lgb:
        classifier_lgb = joblib.load('lgb_Prnw_digi_final_GridS.pkl')
        st.write(Ptext(text, "Lightgbm", classifier_lgb))
        st.write(40*'-')
    if clf_mlp:
        classifier_mlp = joblib.load('MLP_Prnw_digi_final.pkl')
        st.write(Ptext(text, "Neural Network", classifier_mlp))
        st.write(40*'-')
    if clf_nb:
        classifier_nb = joblib.load('NB_Prnw_digi_final.pkl')
        st.write(Ptext(text, "Naive Bayse", classifier_nb))
        st.write(40*'-')
    if clf_lr:
        classifier_lr = joblib.load('lgb_Prnw_digi_final_GridS.pkl')
        st.write(Ptext(text, "Logestic Regression", classifier_lr))
        st.write(40*'-')
    if clf_wve:
        if clf_wve == "Soft Voting":
            classifier_ensemble_S = joblib.load('EnsembleVote_Prnw_digi_final_soft.pkl')
            st.write(Ptext(text, "Ensemble Vote Classifier(Soft Voting)", classifier_ensemble_S))
            st.write(40*'-')
        elif clf_wve == "Hard Voting":
            classifier_ensemble_H = joblib.load('EnsembleVote_Prnw_digi_final_hard.pkl')
            st.write(Ptext(text, "Ensemble Vote Classifier(Hard Voting)", classifier_ensemble_H))
            st.write(40*'-')


elif len(text)<=10 and len(text)>=1:
    print("ورودی قابل قبول نیست")    





























Stopword = ['همچنان', 'مدت', 'چیز', 'سایر', 'جا', 'طی', 'کل', 'کنونی', 'بیرون','های', 'مثلا', 'کامل','ها', 'کاملا','گیرد','شود','است', 'آنکه', 
            'موارد', 'واقعی', 'امور', 'اکنون', 'بطور', 'بخشی', 'تحت', 'چگونه', 'عدم', 'نوعی', 'حاضر', 'وضع', 'مقابل', 'کنار', 'خویش', 'نگاه', 'درون',
            'زمانی', 'بنابراین', 'تو', 'خیلی', 'بزرگ', 'خودش', 'جز', 'اینجا', 'مختلف', 'توسط', 'نوع', 'همچنین', 'آنجا', 'قبل', 'جناح', 'اینها', 'طور', 'شاید',
            'ایشان', 'جهت', 'طریق', 'مانند', 'پیدا', 'ممکن', 'کسانی', 'جای', 'کسی', 'غیر', 'بی', 'قابل', 'درباره', 'جدید', 'وقتی', 'اخیر', 'چرا', 'بیش',
            'روی', 'طرف', 'جریان', 'زیر', 'آنچه', 'البته', 'فقط', 'چیزی', 'چون', 'برابر', 'هنوز', 'بخش', 'زمینه', 'بین', 'بدون', 'استفاد', 'همان', 'نشان',
            'بسیاری', 'بعد', 'عمل', 'روز', 'اعلام', 'چند', 'آنان', 'بلکه', 'امروز', 'تمام', 'بیشتر', 'آیا', 'برخی', 'علیه', 'دیگری', 'ویژه', 'گذشته', 'انجام',
            'حتی', 'داده', 'راه', 'سوی', 'ولی', 'زمان', 'حال', 'تنها', 'بسیار', 'یعنی', 'عنوان', 'همین', 'هبچ', 'پیش', 'وی', 'یکی', 'اینکه', 'وجود'
            , 'شما', 'پس', 'چنین', 'میان', 'مورد', 'چه', 'اگر', 'همه', 'نه', 'دیگر', 'آنها', 'باید', 'هر', 'او', 'ما', 'من', 'تا', 'نیز', 'اما', 
            'یک', 'خود', 'بر', 'یا', 'هم','ای', 'را','دارد', 'این',"می", 'با','دارد','،',',','.', 'آن', 'برای'
            ,'»','«','(',')','؟','?','شده_است','شده','داشت','مکن','آورد','آیند','کرد','آورده_شده_است','دهد','آورند','دهند', 'و', 'در', 'به', 'که', 'از',
            'اندیشیده_اید','کند','هستند','بتواند','برآورد','\u200eشود','شود','\u200e','شوند','خواهند_بود','آمده','یافته_است','ماند','بیاید',
            'رفت','کنید','هستید','دارید','دهید','دارند','بفهمید','داده_شوند','خواهد_گرفت','خواهد_بود','داده_است',
            'خواهد_شد','داده_شده_است','خواهند_داشت','خواهد_داشت','نرسیده_است','نرسیده_است','درخواست','نخواهد_داشت','داده_خواهد_شد','نیستند','نیافته_اند','بگیرید','رفتن','نیستند']
Stopword = set(Stopword)     # حذف استاپ وورد های تکراری


# st.sidebar.header("انتخاب تعداد جمله ی خلاصه از متن")

st.sidebar.write("""
            -----------------------\n
            -----------------------\n
            -----------------------\n
            -----------------------
        """)

st.sidebar.header("Most Used Stopwords")

st.sidebar.dataframe(data=Stopword)


st.sidebar.button("Download")