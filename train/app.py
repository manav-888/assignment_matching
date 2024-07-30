#import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from pathlib import Path
import re


# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file) 
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''  
    return text


# Function to extract features from the text
def extract_features(text):
    keywords = re.findall(r'\b\w+\b', text)
    invoice_number = re.search(r'Invoice Number:\s*(\w+)', text)
    date = re.search(r'Date:\s*(\d{2}/\d{2}/\d{4})', text)
    amounts = re.findall(r'\$\d+.\d{2}', text)
    return {
        'keywords': keywords,
        'invoice_number': invoice_number.group(1) if invoice_number else None,
        'date': date.group(1) if date else None,
        'amounts': amounts
    }

def calculate_cosine_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer().fit_transform([doc1, doc2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def calculate_jaccard_similarity(text1, text2):
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union 


class InvoiceDatabase:
    def __init__(self):
        self.invoices = []

    def add_invoice(self, invoice):
        self.invoices.append(invoice)

    def get_invoices(self):
        return self.invoices
    
def main(input_invoice_path, database):
    input_text = extract_text_from_pdf(input_invoice_path)

    max_similarity = 0
    most_similar_invoice = None

    for invoice in database.get_invoices():
        similarity = calculate_jaccard_similarity(input_text, invoice['text'])
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_invoice = invoice

    return most_similar_invoice, max_similarity


if __name__ == '__main__':

    train_folder = 'train'
    test_folder = 'test'
    
    train_folder_path = Path(train_folder)
    test_folder_path = Path(test_folder)

    train_invoice_paths = list(train_folder_path.glob('*.pdf'))
    test_invoice_paths = list(test_folder_path.glob('*.pdf'))

    db = InvoiceDatabase()
    for invoice_path in train_invoice_paths:
        text = extract_text_from_pdf(invoice_path)
        features = extract_features(text)
        db.add_invoice({'path': invoice_path, 'text': text, 'features': features})

    for test_invoice_path in test_invoice_paths:
        matched_invoice, similarity_score = main(test_invoice_path, db)
        print(f'Test invoice: {test_invoice_path}')
        print(f'Most similar invoice: {matched_invoice["path"]}, Similarity score: {similarity_score}\n')
