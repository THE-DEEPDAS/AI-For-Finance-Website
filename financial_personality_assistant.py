import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class FinancialPersonalityAssistant:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK data download failed: {str(e)}")
        
        self.personality_types = {
            0: "Saver",
            1: "Spender",
            2: "Investor",
            3: "Risk-Averse"
        }
        
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.scaler = StandardScaler()
        
        # key phrases for NLP-based recommendation generation
        self.action_verbs = ['increase', 'reduce', 'maintain', 'consider', 'start', 'optimize']
        self.finance_terms = ['savings', 'investments', 'expenses', 'budget', 'emergency fund']
        
        self.classifier = KMeans(n_clusters=4, random_state=42)

        self.rating_criteria = {
            'savings_discipline': 0.3,
            'investment_diversity': 0.3,
            'spending_control': 0.2,
            'transaction_frequency': 0.2
        }

        # Initialize GPT-2 for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def analyze_transactions(self, transactions):
        if not transactions:
            raise ValueError("No transactions provided")

        df = pd.DataFrame(transactions, columns=['category', 'description', 'amount'])
        
        # Calculate key metrics 
        total_amount = max(df['amount'].sum(), 0.01)  
        category_totals = df.groupby('category')['amount'].sum()
        
        # Safe calculations with default values
        savings_ratio = (category_totals.get('savings', 0) / total_amount) * 100
        investment_ratio = (category_totals.get('investment', 0) / total_amount) * 100
        spending_ratio = (
            (category_totals.get('shopping', 0) + 
             category_totals.get('groceries', 0)) / total_amount * 100
        )

        self.metrics = {
            'savings_rate': round(max(0, savings_ratio), 1),
            'investment_ratio': round(max(0, investment_ratio), 1),
            'top_category': category_totals.idxmax() if not category_totals.empty else 'unknown',
            'problem_category': df.groupby('category').size().idxmax() if not df.empty else 'unknown',
            'reduction': min(30, round(max(0, spending_ratio))),
            'min_savings': max(10, round(20 - savings_ratio)),
            'emergency_fund': 25,
            'months': round(max(0, savings_ratio) / 20, 1),
            'small_amount': round(max(100, total_amount * 0.05), -1),
            'save_amount': min(20, round(10 + max(0, savings_ratio))),
            'investment_suggestion': 'index funds' if investment_ratio < 10 else 'dividend stocks',
            'spending_category': 'entertainment' if spending_ratio < 20 else 'savings',
            'suggested_investment': 'bonds' if investment_ratio > 50 else 'mutual funds'
        }
        
        # Determine personality type s
        if savings_ratio > 30:
            return "Saver"
        elif investment_ratio > 40:
            return "Investor"
        elif spending_ratio > 60:
            return "Spender"
        else:
            return "Risk-Averse"

    def calculate_user_rating(self, df):
        total_amount = df['amount'].sum()
        
        # Calculate rating components
        savings_score = (df[df['category'] == 'savings']['amount'].sum() / total_amount) * 10
        investment_score = (df[df['category'] == 'investment']['amount'].sum() / total_amount) * 10
        
        # Spending control score (lower spending = higher score)
        spending_categories = ['shopping', 'groceries', 'utilities']
        spending_ratio = df[df['category'].isin(spending_categories)]['amount'].sum() / total_amount
        spending_score = (1 - spending_ratio) * 10
        
        # Transaction frequency score
        avg_transaction = df.groupby('category').size().mean()
        frequency_score = min(10, avg_transaction * 2)  # Cap at 10
        
        # Calculate final weighted score
        final_score = (
            savings_score * self.rating_criteria['savings_discipline'] +
            investment_score * self.rating_criteria['investment_diversity'] +
            spending_score * self.rating_criteria['spending_control'] +
            frequency_score * self.rating_criteria['transaction_frequency']
        )
        
        return round(final_score, 1)

    def extract_patterns(self, df):
        # Extract spending patterns
        patterns = {
            'total_spent': df['amount'].sum(),
            'category_ratios': df.groupby('category')['amount'].sum() / df['amount'].sum(),
            'transaction_frequency': df.groupby('category').size(),
            'average_transaction': df.groupby('category')['amount'].mean(),
            'spending_volatility': df.groupby('category')['amount'].std().fillna(0)
        }
        return patterns

    def generate_nlp_recommendation(self, category, metrics, trends):
        """Generate dynamic recommendation using NLP"""
        # Create context for the model
        context = f"Financial advice about {category}: "
        if trends['trend'] > 0:
            context += f"spending increased by {trends['change_percent']}% "
        else:
            context += f"spending decreased by {abs(trends['change_percent'])}% "
        
        context += f"Currently {metrics['ratio']:.1f}% of total spending. "
        
        # Generate text using GPT-2
        inputs = self.tokenizer.encode(context, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean and format the generated text
        return self.clean_generated_text(generated_text)
        
    def analyze_category_trends(self, df, category):
        """Analyze spending trends for a category"""
        category_data = df[df['category'] == category]
        if len(category_data) < 2:
            return {'trend': 0, 'change_percent': 0}
            
        total_spent = category_data['amount'].sum()
        avg_transaction = category_data['amount'].mean()
        
        # Calculate trend
        trend = np.polyfit(range(len(category_data)), category_data['amount'].values, 1)[0]
        change_percent = (trend / avg_transaction) * 100 if avg_transaction != 0 else 0
        
        return {
            'trend': trend,
            'change_percent': round(change_percent, 1)
        }
        
    def clean_generated_text(self, text):
        """Clean and format the generated text"""
        sentences = sent_tokenize(text)
        # Take the most relevant sentence
        cleaned_text = sentences[0].strip()
        # Ensure it's advice-oriented
        if not any(word in cleaned_text.lower() for word in ['should', 'consider', 'try', 'recommend']):
            cleaned_text = f"Consider: {cleaned_text}"
        return cleaned_text

    def generate_dynamic_recommendations(self, df, personality_type):
        recommendations = []
        patterns = self.extract_patterns(df)
        categories = df['category'].unique()
        
        # Generate category-specific recommendations
        for category in categories:
            trends = self.analyze_category_trends(df, category)
            metrics = {
                'ratio': (patterns['category_ratios'].get(category, 0) * 100),
                'avg_amount': patterns['average_transaction'].get(category, 0)
            }
            
            rec = self.generate_nlp_recommendation(category, metrics, trends)
            if rec and len(recommendations) < 5:
                recommendations.append(rec)
        
        # If we need more recommendations
        while len(recommendations) < 5:
            # Generate general financial advice
            context = f"Financial advice for a {personality_type.lower()} personality: "
            inputs = self.tokenizer.encode(context, return_tensors='pt')
            outputs = self.model.generate(
                inputs, 
                max_length=75,
                temperature=0.8,
                num_return_sequences=1,
                do_sample=True
            )
            advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_advice = self.clean_generated_text(advice)
            if cleaned_advice not in recommendations:
                recommendations.append(cleaned_advice)
                
        return recommendations[:5]

    def process_user_data(self, transactions):
        try:
            df = pd.DataFrame(transactions, columns=['category', 'description', 'amount'])
            personality = self.analyze_transactions(transactions)
            recommendations = self.generate_dynamic_recommendations(df, personality)
            user_rating = self.calculate_user_rating(df)
            
            return {
                "personality": personality,
                "recommendations": recommendations,
                "user_rating": user_rating
            }
        except Exception as e:
            return {
                "personality": "Unknown",
                "recommendations": [f"Error analyzing transactions: {str(e)}"],
                "user_rating": 0.0
            }
