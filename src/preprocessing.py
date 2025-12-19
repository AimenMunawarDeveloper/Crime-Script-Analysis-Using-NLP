import pandas as pd
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    nlp = None


class TextPreprocessor:
    
    def __init__(self):
        self.acronym_dict = {
            'ICA': 'immigration and checkpoints authority',
            'ID': 'identity',
            'DBS': 'dbs bank',
            'FB': 'facebook',
            'SG': 'singapore',
            'UK': 'united kingdom',
            'NRIC': 'identity number',
            'IC': 'identity number',
            'I/C': 'identity number',
            'HQ': 'headquarters',
            'MOM': 'ministry of manpower',
            'POSB': 'posb bank',
            'MOH': 'ministry of health',
            'OCBC': 'ocbc bank',
            'CMB': 'cmb bank',
            'SPF': 'singapore police force',
            'IRAS': 'inland revenue authority of singapore',
            'UOB': 'uob bank',
            'IG': 'instagram',
            'HP': 'handphone',
            'HK': 'hong kong',
            'KL': 'kuala lumpur',
            'PM': 'private message',
            'MRT': 'mass rapid transit train',
            'DOB': 'date of birth',
            'ATM': 'automated teller machine',
            'MAS': 'monetary authority of singapore',
            'PRC': 'people republic of china',
            'USS': 'universal studios singapore',
            'MIA': 'missing in action',
            'GST': 'goods and services tax',
            'CIMB': 'cimb bank',
            'HSBC': 'hsbc bank',
            'MBS': 'marina_bay_sands',
            'LTD': 'limited',
            'ASAP': 'as soon as possible',
            'IBAN': 'international bank account number',
            'HR': 'human resource',
            'AMK': 'ang mo kio',
            'CID': 'criminal investigation department',
            'PTE': 'private',
            'OTP': 'one time password',
            'WA': 'whatsapp',
            'PC': 'personal computer',
            'ACRA': 'accounting and corporate regulatory authority',
            'CPF': 'central provident fund',
            'ISD': 'internal security department',
            'WP': 'work permit',
            'OKC': 'okcupid',
            'HDB': 'housing development board',
            'NPC': 'neighbourhood police centre',
            'MOP': 'member of public',
            'MOPS': 'members of public',
            'IMO': 'in my opinion',
            'ISP': 'internet service provider',
            'IMDA': 'infocomm media development authority',
            'CB': 'circuit breaker',
            'MINLAW': 'ministry of law',
            'LMAO': 'laugh my ass off',
            'AKA': 'also known as',
            'BF': 'boyfriend',
            'W/O': 'without',
            'MOF': 'ministry of finance'
        }
        
        self.spellcheck_dict = {
            'acct': 'account',
            'acc': 'account',
            'a/c': 'account',
            'blk': 'block',
            'alot': 'a lot',
            'abit': 'a bit',
            'watsapp': 'whatsapp',
            'whatapps': 'whatsapp',
            'whatapp': 'whatsapp',
            'wadsapp': 'whatsapp',
            'watapps': 'whatsapp',
            'whatsapps': 'whatsapp',
            'whats app': 'whatsapp',
            'whatsaap': 'whatsapp',
            'whatsap': 'whatsapp',
            'whattsapp': 'whatsapp',
            'whattapp': 'whatsapp',
            'whataspp': 'whatsapp',
            'whastapp': 'whatsapp',
            'whatsapphe': 'whatsapp',
            'abt': 'about',
            'recieved': 'received',
            'recieve': 'receive',
            'hv': 'have',
            'amt': 'amount',
            'mths': 'months',
            'gf': 'girlfriend',
            'msia': 'malaysia',
            'tranfer': 'transfer',
            'trans': 'transfer',
            'trf': 'transfer',
            'becareful': 'be careful',
            'frm': 'from',
            'msgs': 'messages',
            'msg': 'message',
            'plz': 'please',
            'pls': 'please',
            'harrass': 'harass',
            'sintel': 'singtel',
            'ard': 'around',
            'wk': 'week',
            'fyi': 'for your information',
            'govt': 'government',
            'gov': 'government',
            'thru': 'through',
            'assent': 'accent',
            'dun': 'do not',
            'nv': 'never',
            'sing-tel': 'singtel',
            'insta': 'instagram',
            'sg': 'singapore',
            'payapl': 'paypal',
            'carousel': 'carousell',
            'tix': 'tickets',
            'mandrain': 'mandarin',
            'admin': 'administrative',
            'bz': 'busy',
            'daugter': 'daughter',
            'cos': 'because',
            'bcos': 'because',
            'I-banking': 'internet banking',
            'intl': 'international',
            'shoppe': 'shopee',
            'tis': 'this',
            'docs': 'documents',
            'doc': 'document',
            'ytd': 'yesterday',
            'tmr': 'tomorrow',
            'mon': 'monday',
            'tue': 'tuesday',
            'tues': 'tuesday',
            'wed': 'wednesday',
            'thu': 'thursday',
            'thur': 'thursday',
            'thurs': 'thursday',
            'fri': 'friday',
            'wikipeida': 'wikipedia',
            'juz': 'just',
            'impt': 'important',
            'transger': 'transfer',
            'suspicios': 'suspicious',
            'suspicius': 'suspicious',
            'suspicous': 'suspicious',
            'suspecious': 'suspicious',
            'suspision': 'suspicion',
            'nvr': 'never',
            'instagam': 'instagram',
            'instagramm': 'instagram',
            "s'pore": "singapore",
            'polive': 'police',
            'linkein': 'linkedin',
            'messanger': 'messenger',
            'scammmer': 'scammer',
            'laywer': 'lawyer',
            'dunno': 'do not know',
            'tidner': 'tinder',
            'rcvd': 'received',
            'infomed': 'informed',
            'informaing': 'informing',
            'knowldge': 'knowledge'
        }
    
    def remove_url(self, text: str) -> str:
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))"
        text = re.sub(regex, "url_link", text)
        return text
    
    def decontract(self, phrase: str) -> str:
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"let\'s", "let us", phrase)
        
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"let\s", "let us", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        
        return phrase
    
    def remove_punct(self, text: str) -> str:
        punctuation = "``-±!@#$%^&*()+?:;""'<>"
        for c in text:
            if c in punctuation:
                text = text.replace(c, "").replace('/', ' ').replace('`', "").replace('"', '')
        return text
    
    def unabbreviate(self, text: str) -> str:
        x = word_tokenize(text)
        for index, token in enumerate(x):
            for k, v in self.acronym_dict.items():
                if token == k or token == k.lower():
                    x[index] = v
                    break
        return ' '.join(x).replace(" .", ".").replace(" ,", ",")
    
    def correct_misspelled_words(self, text: str) -> str:
        x = word_tokenize(text)
        for index, token in enumerate(x):
            for k, v in self.spellcheck_dict.items():
                if token == k:
                    x[index] = v
                    break
        return ' '.join(x).replace(' .', '.').replace(' ,', ',').replace('< ', '<').replace(' >', '>')
    
    def remove_stopwords(self, text_string: str) -> str:
        word_list = [word for word in word_tokenize(text_string) 
                    if word not in set(stopwords.words('english'))]
        text = ' '.join(word_list).replace(' .', '').replace(' ,', '').replace('< ', '<').replace(' >', '>')
        return text
    
    def lemmatise(self, text_string: str) -> str:
        if nlp is None:
            return text_string
        list_of_tokens = [token.lemma_ for token in nlp(text_string)]
        text = ' '.join(list_of_tokens).replace('< ', '<').replace(' >', '>')
        return text
    
    def preprocess(self, text: str) -> str:
        text = text.encode('ascii', 'ignore').decode('utf-8')
        
        text = self.remove_url(text.replace('\n', ' '))
        
        text = re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', text)
        
        text = re.sub(r'\d+', '', text)
        
        text = self.decontract(text)
        
        text = text.lower().strip().replace("'s", "")
        
        text = self.remove_punct(text)
        
        text = self.unabbreviate(text)
        
        text = self.correct_misspelled_words(text)
        
        return text
    
    def load_dataset(self, filepath: str, text_column: str = 'incident_description', 
                    id_column: str = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            
            first_col = df.columns[0]
            if '<<<<<<< HEAD' in str(first_col) or '<<<<<<< HEAD' in str(df.iloc[0, 0] if len(df) > 0 else ''):
                print("Warning: Detected Git merge conflict markers. Attempting to clean the file...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                header_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(',submission_id') or line.strip().startswith('submission_id'):
                        header_idx = i
                        break
                
                df = pd.read_csv(filepath, skiprows=header_idx)
            
            if text_column not in df.columns:
                possible_text_columns = ['incident_description', 'description', 'text', 'incident', 'story', 'report']
                for col in possible_text_columns:
                    if col in df.columns:
                        text_column = col
                        print(f"Auto-detected text column: {text_column}")
                        break
                
                if text_column not in df.columns:
                    print(f"Available columns: {df.columns.tolist()}")
                    raise ValueError(f"Text column '{text_column}' not found. Available columns: {df.columns.tolist()}")
            
            print(f"Loaded dataset with {len(df)} records from {filepath}")
            print(f"Using text column: {text_column}")
            self.detected_text_column = text_column
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'incident_description',
                          id_column: str = None) -> pd.DataFrame:
        df = df.copy()
        
        df['preprocessed_text'] = ""
        df['stopwords_removed'] = ""
        df['lemmatised'] = ""
        
        print("Preprocessing text data...")
        for index, row in df.iterrows():
            preprocessed = self.preprocess(str(row[text_column]))
            df.at[index, 'preprocessed_text'] = preprocessed
            
            stopwords_removed = self.remove_stopwords(preprocessed)
            df.at[index, 'stopwords_removed'] = stopwords_removed
            
            lemmatised = self.lemmatise(stopwords_removed)
            df.at[index, 'lemmatised'] = lemmatised
            
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(df)} records...")
        
        df['len_preprocessed_text'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
        df['len_lemmatised'] = df['lemmatised'].apply(lambda x: len(x.split()))
        
        print(f"Preprocessing complete! Processed {len(df)} records.")
        return df
    
    def save_preprocessed_data(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)
        print(f"Preprocessed data saved to {filepath}")
