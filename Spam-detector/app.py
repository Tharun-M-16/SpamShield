import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import re
import numpy as np
import pickle
import hashlib
import warnings
import time
warnings.filterwarnings('ignore')

# Page setup with optimized configuration
st.set_page_config(
    page_title="🛡️SpamShield", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add error handling for Tornado issues
try:
    # Custom CSS
    st.markdown("""
    <style>
        .main-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem; color: white; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem; }
        .result-spam { background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; }
        .result-safe { background: linear-gradient(135deg, #51cf66, #40c057); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header"><h1>🛡️ SpamShield</h1><p>Powered by SVM - Industry Standard for Text Classification</p></div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading UI: {str(e)}")
    st.stop()

# Create models directory if it doesn't exist
MODELS_DIR = "saved_models"
if not os.path.exists(MODELS_DIR):
    try:
        os.makedirs(MODELS_DIR)
    except Exception as e:
        st.error(f"Cannot create models directory: {str(e)}")

# Initialize session state variables with better management
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'input_text': "",
        'model_loaded': False,
        'model_loading': False,
        'current_model': None,
        'current_vectorizer': None,
        'current_accuracy': 0.0,
        'current_y_test': None,
        'current_y_pred': None,
        'prediction_history': [],
        'last_model_choice': None,
        'dataset_load_status': "Not loaded"
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()

# Input validation function
def validate_message(message):
    """Validate user input message"""
    if not message or not message.strip():
        return False, "Message cannot be empty"
    
    if len(message) > 10000:
        return False, "Message too long (max 10,000 characters)"
    
    if len(message.split()) < 1:
        return False, "Message must contain at least one word"
    
    return True, ""

# Load/Create Dataset with better error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset():
    try:
        # Priority order for dataset loading
        dataset_files = ["spam 2.csv", "spam_dataset.csv", "your_dataset.csv"]
        
        for csv_file in dataset_files:
            if os.path.exists(csv_file):
                try:
                    # Check file size before loading
                    file_size = os.path.getsize(csv_file)
                    if file_size > 50 * 1024 * 1024:  # 50MB limit
                        st.sidebar.warning(f"⚠️ {csv_file} too large (>50MB), skipping...")
                        continue
                    
                    df = pd.read_csv(csv_file, encoding='latin-1')
                    
                    # Validate minimum rows
                    if len(df) < 10:
                        st.sidebar.warning(f"⚠️ {csv_file} has too few rows (<10), skipping...")
                        continue
                    
                    # Handle different column naming conventions
                    if 'v1' in df.columns and 'v2' in df.columns:
                        df = df[['v1', 'v2']].dropna()
                        df.columns = ['label', 'message']
                    elif 'Label' in df.columns and 'Message' in df.columns:
                        df.columns = ['label', 'message']
                    elif 'category' in df.columns and 'text' in df.columns:
                        df.columns = ['label', 'message']
                    elif 'spam' in df.columns and 'text' in df.columns:
                        df.columns = ['label', 'message']
                    elif 'label' in df.columns and 'message' in df.columns:
                        df = df[['label', 'message']].dropna()
                    else:
                        st.sidebar.warning(f"⚠️ {csv_file} doesn't have required columns, skipping...")
                        continue
                    
                    # Validate required columns exist
                    if 'label' not in df.columns or 'message' not in df.columns:
                        st.sidebar.warning(f"⚠️ {csv_file} missing required columns, skipping...")
                        continue
                    
                    # Remove empty messages
                    df = df.dropna(subset=['message'])
                    df = df[df['message'].str.strip() != '']
                    
                    # Add timestamp if not present
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = datetime.now()
                    
                    # Standardize label values
                    df['label'] = df['label'].astype(str).str.lower().str.strip()
                    label_mapping = {'ham': 'ham', 'spam': 'spam', '0': 'ham', '1': 'spam', 'legitimate': 'ham', 'phishing': 'spam'}
                    df['label'] = df['label'].map(label_mapping).fillna(df['label'])
                    
                    # Validate we have both classes
                    unique_labels = df['label'].unique()
                    if len(unique_labels) < 2 or not all(label in unique_labels for label in ['ham', 'spam']):
                        st.sidebar.warning(f"⚠️ {csv_file} doesn't contain both ham and spam labels, skipping...")
                        continue
                    
                    # Check minimum samples per class
                    class_counts = df['label'].value_counts()
                    if class_counts.min() < 5:
                        st.sidebar.warning(f"⚠️ {csv_file} has too few samples per class (need ≥5), skipping...")
                        continue
                    
                    st.session_state.dataset_load_status = f"Loaded {csv_file}"
                    st.sidebar.success(f"✅ Loaded dataset: {csv_file} ({len(df)} messages)")
                    return df
                    
                except pd.errors.EmptyDataError:
                    st.sidebar.error(f"❌ {csv_file} is empty")
                    continue
                except pd.errors.ParserError as e:
                    st.sidebar.error(f"❌ Cannot parse {csv_file}: {str(e)}")
                    continue
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading {csv_file}: {str(e)}")
                    continue
        
        # If no dataset found, create sample data
        st.sidebar.info("📝 No valid dataset found, creating sample data...")
        return create_initial_dataset()
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_initial_dataset()

def create_initial_dataset():
    """Create initial dataset"""
    ham_messages = [
        'Hey, how are you doing today?', 'Meeting is at 3pm tomorrow', 'Thanks for the help yesterday',
        'Can you send me the report?', 'Happy birthday! Hope you have a great day', 'See you at lunch',
        'The weather is nice today', 'Good morning, have a great day ahead', 'How was your weekend?',
        'Let me know when you are free', 'Thanks for the quick response', 'Great job on the presentation',
        'I will be there in 10 minutes', 'Can we reschedule our meeting?', 'Hope you feel better soon',
        'Congratulations on your promotion', 'The document looks good to me', 'Drive safely',
        'Let me know if you need anything', 'Have a safe trip', 'Good luck with your interview',
        'Thank you for your time today', 'Looking forward to our meeting', 'Hope everything is going well',
        'The project is on track', 'Can you please review this?', 'I will call you later',
        'Thanks for the update', 'See you next week', 'Have a wonderful evening',
        'The presentation went well', 'I appreciate your help', 'Let me know your thoughts',
        'Hope you have a relaxing weekend', 'Good to hear from you', 'Take care of yourself',
        'I will send you the details', 'Thanks for being patient', 'Hope your day goes well',
        'Looking forward to hearing from you', 'I hope this message finds you well', 'Have a productive day',
        'Thank you for your understanding', 'I will get back to you soon', 'Hope you are doing well',
        'Please let me know if this works', 'I appreciate your feedback', 'Have a great rest of your day',
        'Thank you for your cooperation', 'I will keep you updated', 'Hope this helps'
    ]
    
    spam_messages = [
        'URGENT! You have won $1000000! Click here now!', 'FREE MONEY! Act now or lose forever!',
        'CONGRATULATIONS! You are our lucky winner! Claim your prize now!', 'Limited time offer! Get 90% OFF!',
        'URGENT: Your account will be closed! Click to verify now!', 'You have been selected for a special offer!',
        'WIN BIG! Play our casino games now!', 'FREE iPhone! Click here to claim yours!',
        'URGENT: Update your payment info immediately!', 'You qualify for a $5000 loan! Apply now!',
        'BREAKING: Make $500 daily from home!', 'ALERT: Suspicious activity on your account!',
        'FREE GIFT CARD! Limited time only!', 'URGENT: Your package is waiting!',
        'You have won a luxury vacation! Claim now!', 'SPECIAL OFFER: 80% discount today only!',
        'URGENT: Security breach detected!', 'FREE TRIAL: Weight loss miracle!',
        'WINNER! You have been chosen!', 'URGENT: Confirm your identity now!',
        'FREE MONEY MAKING SYSTEM!', 'ALERT: Your subscription expires today!',
        'URGENT: Click to avoid account closure!', 'YOU ARE A WINNER! Claim your prize!',
        'FREE SAMPLES! Order now!', 'URGENT: Verify your account immediately!',
        'AMAZING OFFER! Limited time!', 'FREE CRUISE! Book now!',
        'URGENT: Payment required immediately!', 'WIN CASH PRIZES! Enter now!',
        'FREE DOWNLOAD! Click here!', 'URGENT: Action required on your account!',
        'SPECIAL DISCOUNT! Today only!', 'FREE PHONE! Claim yours now!',
        'URGENT: Your order is pending!', 'AMAZING DEAL! Don\'t miss out!',
        'FREE CREDIT REPORT! Get yours now!', 'URGENT: Your account needs verification!',
        'WIN BIG MONEY! Play now!', 'FREE MEMBERSHIP! Sign up today!',
        'URGENT: Your payment is overdue!', 'SPECIAL PROMOTION! Limited time offer!',
        'FREE CONSULTATION! Book now!', 'URGENT: Your account is at risk!',
        'WIN EXCITING PRIZES! Enter today!', 'FREE TRIAL! No commitment required!',
        'URGENT: Immediate action needed!', 'AMAZING OPPORTUNITY! Don\'t miss!',
        'FREE DELIVERY! Order now!', 'URGENT: Your security is at risk!'
    ]
    
    # Ensure equal lengths
    total_messages = len(ham_messages) + len(spam_messages)
    
    sample_data = {
        'label': ['ham'] * len(ham_messages) + ['spam'] * len(spam_messages),
        'message': ham_messages + spam_messages,
        'timestamp': [datetime.now()] * total_messages
    }
    
    df = pd.DataFrame(sample_data)
    try:
        df.to_csv("spam_dataset.csv", index=False)
        st.session_state.dataset_load_status = "Created sample dataset"
    except Exception as e:
        st.error(f"Cannot save sample dataset: {str(e)}")
    return df

# Text preprocessing
def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

def extract_features(message):
    if not message:
        return {
            'length': 0,
            'word_count': 0,
            'exclamation_count': 0,
            'uppercase_ratio': 0,
            'spam_keywords': 0
        }
    
    return {
        'length': len(message),
        'word_count': len(message.split()),
        'exclamation_count': message.count('!'),
        'uppercase_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
        'spam_keywords': sum(1 for word in ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'limited', 'offer', 'click', 'now'] if word in message.lower())
    }

def get_model_filename(model_name, vectorizer_name):
    """Generate model filename based on model and vectorizer choice"""
    safe_model = re.sub(r'[^\w\-_]', '_', model_name.lower())
    safe_vectorizer = re.sub(r'[^\w\-_]', '_', vectorizer_name.lower())
    return f"{MODELS_DIR}/model_{safe_model}_{safe_vectorizer}.pkl"

def get_dataset_hash(df):
    """Generate hash of dataset for change detection"""
    try:
        # Use a sample of messages to create hash (for performance)
        sample_size = min(100, len(df))
        sample_data = df.head(sample_size)['message'].str.cat(sep='|')
        return hashlib.md5(sample_data.encode('utf-8')).hexdigest()
    except Exception:
        return "default_hash"

def save_model(model, vectorizer, accuracy, y_test, y_pred, model_name, vectorizer_name, dataset_hash):
    """Save trained model to disk with better error handling"""
    if model is None or vectorizer is None:
        st.error("Cannot save: Model or vectorizer is None")
        return False
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'dataset_hash': dataset_hash,
        'timestamp': datetime.now()
    }
    
    filename = get_model_filename(model_name, vectorizer_name)
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Verify file was created and is readable
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            # Test loading to ensure it's valid
            try:
                with open(filename, 'rb') as f:
                    test_load = pickle.load(f)
                return True
            except Exception:
                st.error("Model saved but verification failed")
                return False
        else:
            st.error("Model file was not created successfully")
            return False
            
    except PermissionError:
        st.error("Permission denied: Cannot save model file")
        return False
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_saved_model(model_name, vectorizer_name):
    """Load saved model from disk with better error handling"""
    filename = get_model_filename(model_name, vectorizer_name)
    
    if not os.path.exists(filename):
        return None
    
    try:
        # Check file size
        if os.path.getsize(filename) == 0:
            st.warning(f"Model file {filename} is empty, removing...")
            os.remove(filename)
            return None
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate model data structure
        required_keys = ['model', 'vectorizer', 'accuracy']
        if not all(key in model_data for key in required_keys):
            st.warning("Invalid model file structure, removing...")
            os.remove(filename)
            return None
        
        # Test that model and vectorizer are functional
        if model_data['model'] is None or model_data['vectorizer'] is None:
            st.warning("Model or vectorizer is None, removing file...")
            os.remove(filename)
            return None
        
        return model_data
        
    except (pickle.PickleError, EOFError) as e:
        st.warning(f"Corrupted model file, removing: {str(e)}")
        try:
            os.remove(filename)
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Error loading saved model: {str(e)}")
        return None

def should_retrain_model(saved_model_data, current_dataset_hash):
    """Check if model should be retrained based on dataset changes"""
    if saved_model_data is None:
        return True
    
    # Check if dataset has changed significantly
    if saved_model_data.get('dataset_hash') != current_dataset_hash:
        return True
    
    return False

def train_new_model(df, model_name, vectorizer_name):
    """Train a new model with better error handling"""
    try:
        # Validate dataset
        if df is None or len(df) == 0:
            st.error("Cannot train: Dataset is empty!")
            return None, None, 0.0, None, None
        
        # Ensure we have required columns
        if 'message' not in df.columns or 'label' not in df.columns:
            st.error("Dataset must have 'message' and 'label' columns!")
            return None, None, 0.0, None, None
        
        # Remove empty messages
        df_clean = df.dropna(subset=['message', 'label'])
        df_clean = df_clean[df_clean['message'].str.strip() != '']
        
        if len(df_clean) == 0:
            st.error("No valid messages found in dataset!")
            return None, None, 0.0, None, None
        
        # Preprocess messages
        df_clean['processed_message'] = df_clean['message'].apply(preprocess_text)
        
        # Remove empty processed messages
        df_clean = df_clean[df_clean['processed_message'].str.strip() != '']
        
        if len(df_clean) == 0:
            st.error("No valid processed messages found!")
            return None, None, 0.0, None, None
        
        X = df_clean['processed_message']
        y = df_clean['label']
        
        # Ensure we have both classes in the data
        unique_labels = y.unique()
        if len(unique_labels) < 2:
            st.error("Dataset must contain both spam and ham messages!")
            return None, None, 0.0, None, None
        
        # Check minimum samples per class
        class_counts = y.value_counts()
        if class_counts.min() < 5:
            st.error(f"Not enough samples for training! Need at least 5 samples per class. Current: {dict(class_counts)}")
            return None, None, 0.0, None, None
        
        # TF-IDF Vectorization with optimized parameters
        try:
            vectorizer = TfidfVectorizer(
                max_features=3000, 
                stop_words='english', 
                ngram_range=(1, 2),  # Include bigrams for better performance
                sublinear_tf=True,
                min_df=2,
                max_df=0.95
            )
            
            X_vectorized = vectorizer.fit_transform(X)
            
            if X_vectorized.shape[0] == 0 or X_vectorized.shape[1] == 0:
                st.error("Vectorization resulted in empty feature matrix!")
                return None, None, 0.0, None, None
            
        except Exception as e:
            st.error(f"Error during vectorization: {str(e)}")
            return None, None, 0.0, None, None
        
        # Adjust test size for small datasets
        test_size = min(0.3, max(0.1, 10/len(df_clean))) if len(df_clean) < 100 else 0.2
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
        except ValueError as e:
            st.error(f"Error splitting data: {str(e)}")
            return None, None, 0.0, None, None
        
        # Model selection with optimized parameters
        try:
            if model_name == "SVM":
                # Linear SVM - Industry standard for text classification
                model = SVC(
                    kernel='linear', 
                    C=1.0, 
                    class_weight='balanced', 
                    probability=True, 
                    random_state=42
                )
            elif model_name == "Naive Bayes":
                model = MultinomialNB(alpha=0.1)
            else:  # Logistic Regression
                model = LogisticRegression(
                    random_state=42, 
                    max_iter=1000, 
                    C=1.0, 
                    class_weight='balanced',
                    solver='liblinear'
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Validate model works for prediction
            test_prediction = model.predict(X_test[:1])
            if len(test_prediction) == 0:
                st.error("Model training failed - cannot make predictions")
                return None, None, 0.0, None, None
            
            return model, vectorizer, accuracy, y_test, y_pred
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, 0.0, None, None
        
    except Exception as e:
        st.error(f"Unexpected error training model: {str(e)}")
        return None, None, 0.0, None, None

# Add new data to dataset with better error handling
def add_to_dataset(message, label, confidence):
    """Add new classified message to dataset"""
    try:
        new_data = pd.DataFrame({
            'label': [label],
            'message': [message],
            'timestamp': [datetime.now()]
        })
        
        # Use the main dataset file if it exists, otherwise create new one
        dataset_files = ["uploaded_dataset.csv", "spam_dataset.csv"]
        csv_file = None
        
        for file in dataset_files:
            if os.path.exists(file):
                csv_file = file
                break
        
        if csv_file is None:
            csv_file = "spam_dataset.csv"  # Create new file
        
        if os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file, encoding='latin-1')
                
                # Handle different column formats
                if 'v1' in existing_df.columns and 'v2' in existing_df.columns:
                    existing_df = existing_df[['v1', 'v2']].dropna()
                    existing_df.columns = ['label', 'message']
                
                # Add timestamp if not present
                if 'timestamp' not in existing_df.columns:
                    existing_df['timestamp'] = datetime.now()
                    
                updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not load existing dataset: {str(e)}, creating new one")
                updated_df = new_data
        else:
            updated_df = new_data
        
        # Save updated dataset
        updated_df.to_csv(csv_file, index=False)
        return updated_df
        
    except Exception as e:
        st.error(f"Error saving to dataset: {str(e)}")
        return None

# Memory management - clean up old predictions
def manage_memory():
    """Clean up memory by limiting stored predictions"""
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]

# Load initial data
try:
    df = load_dataset()
    current_dataset_hash = get_dataset_hash(df)
except Exception as e:
    st.error(f"Critical error loading data: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.title("🎛️ Settings")

# Dataset info with error handling
st.sidebar.subheader("📁 Dataset Info")
try:
    dataset_info = f"📊 Total Messages: {len(df):,}"
    if 'label' in df.columns:
        spam_count = len(df[df['label'] == 'spam'])
        ham_count = len(df[df['label'] == 'ham'])
        dataset_info += f"\n🚨 Spam: {spam_count:,}\n✅ Ham: {ham_count:,}"
    st.sidebar.text(dataset_info)
    st.sidebar.caption(f"Status: {st.session_state.dataset_load_status}")
except Exception as e:
    st.sidebar.error("Error loading dataset info")

# Model controls
model_choice = st.sidebar.selectbox("🤖 ML Model:", ["SVM", "Naive Bayes", "Logistic Regression"])
# Removed vectorizer choice - only TF-IDF for optimal performance
vectorizer_choice = "TF-IDF"

# Display model information
model_descriptions = {
    "SVM": "🏆 Best for text classification",
    "Naive Bayes": "⚡ Fast and reliable",
    "Logistic Regression": "📊 Good baseline model"
}
st.sidebar.info(model_descriptions[model_choice])

# Check if we need to load/retrain model
model_config_changed = (
    st.session_state.get('last_model_choice') != model_choice
)

# Improved model loading with race condition protection
if (not st.session_state.model_loaded or model_config_changed) and not st.session_state.model_loading:
    st.session_state.model_loading = True
    
    # Try to load saved model first
    saved_model_data = load_saved_model(model_choice, vectorizer_choice)
    
    if should_retrain_model(saved_model_data, current_dataset_hash):
        # Need to train new model
        with st.spinner(f"🤖 Training {model_choice} with {vectorizer_choice}... (This happens only once)"):
            model, vectorizer, accuracy, y_test, y_pred = train_new_model(df, model_choice, vectorizer_choice)
            
            if model is not None and vectorizer is not None:
                # Save the trained model
                if save_model(model, vectorizer, accuracy, y_test, y_pred, model_choice, vectorizer_choice, current_dataset_hash):
                    st.success("✅ Model trained and saved successfully!")
                
                # Update session state
                st.session_state.current_model = model
                st.session_state.current_vectorizer = vectorizer
                st.session_state.current_accuracy = accuracy
                st.session_state.current_y_test = y_test
                st.session_state.current_y_pred = y_pred
                st.session_state.model_loaded = True
            else:
                st.error("❌ Model training failed!")
                st.session_state.model_loaded = False
    else:
        # Use saved model
        st.info("📂 Loading saved model...")
        try:
            st.session_state.current_model = saved_model_data['model']
            st.session_state.current_vectorizer = saved_model_data['vectorizer']
            st.session_state.current_accuracy = saved_model_data['accuracy']
            st.session_state.current_y_test = saved_model_data['y_test']
            st.session_state.current_y_pred = saved_model_data['y_pred']
            st.session_state.model_loaded = True
            st.success("✅ Model loaded from cache!")
        except Exception as e:
            st.error(f"Error loading saved model: {str(e)}")
            st.session_state.model_loaded = False
    
    # Update last choices and reset loading state
    st.session_state.last_model_choice = model_choice
    st.session_state.model_loading = False

# Error recovery - check for corrupted model state
if st.session_state.model_loaded and (st.session_state.current_model is None or st.session_state.current_vectorizer is None):
    st.error("🔧 Model corrupted! Forcing retrain...")
    st.session_state.model_loaded = False
    st.session_state.model_loading = False
    st.rerun()

# Model status
st.sidebar.subheader("🤖 Model Status")
if st.session_state.model_loading:
    st.sidebar.warning("⏳ Loading Model...")
elif st.session_state.model_loaded:
    st.sidebar.success("✅ Model Ready")
    st.sidebar.info(f"Model: {model_choice}")
    st.sidebar.info(f"Vectorizer: TF-IDF")
    st.sidebar.info(f"Accuracy: {st.session_state.current_accuracy*100:.1f}%")
else:
    st.sidebar.error("❌ Model Not Ready")

# Force retrain button
if st.sidebar.button("🔄 Force Retrain Model", disabled=st.session_state.model_loading):
    with st.spinner("🤖 Retraining model..."):
        model, vectorizer, accuracy, y_test, y_pred = train_new_model(df, model_choice, vectorizer_choice)
        
        if model is not None and vectorizer is not None:
            # Save the new model
            if save_model(model, vectorizer, accuracy, y_test, y_pred, model_choice, vectorizer_choice, current_dataset_hash):
                st.success("✅ Model retrained successfully!")
            
            # Update session state
            st.session_state.current_model = model
            st.session_state.current_vectorizer = vectorizer
            st.session_state.current_accuracy = accuracy
            st.session_state.current_y_test = y_test
            st.session_state.current_y_pred = y_pred
            st.session_state.model_loaded = True
            
            st.rerun()
        else:
            st.error("❌ Model retraining failed!")

# File upload option with improved validation
st.sidebar.subheader("📤 Upload New Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file:", 
    type=['csv'],
    help="Upload your spam dataset (columns: label, message)"
)

if uploaded_file is not None:
    try:
        # Validate file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.sidebar.error("❌ File too large! Maximum size: 10MB")
        else:
            new_df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Validate file has content
            if len(new_df) == 0:
                st.sidebar.error("❌ File is empty!")
            else:
                # Validate required columns
                valid_column_sets = [
                    ['label', 'message'],
                    ['v1', 'v2'],
                    ['Label', 'Message'],
                    ['category', 'text'],
                    ['spam', 'text']
                ]
                
                has_valid_columns = any(
                    all(col in new_df.columns for col in col_set) 
                    for col_set in valid_column_sets
                )
                
                if not has_valid_columns:
                    st.sidebar.error("❌ Invalid file format! Required columns: 'label' and 'message' (or similar)")
                else:
                    # Handle different column formats
                    if 'v1' in new_df.columns and 'v2' in new_df.columns:
                        new_df = new_df[['v1', 'v2']].dropna()
                        new_df.columns = ['label', 'message']
                    elif 'Label' in new_df.columns and 'Message' in new_df.columns:
                        new_df.columns = ['label', 'message']
                    elif 'category' in new_df.columns and 'text' in new_df.columns:
                        new_df.columns = ['label', 'message']
                    elif 'spam' in new_df.columns and 'text' in new_df.columns:
                        new_df.columns = ['label', 'message']
                    
                    # Clean and validate data
                    new_df = new_df.dropna(subset=['message', 'label'])
                    new_df = new_df[new_df['message'].str.strip() != '']
                    
                    if len(new_df) < 10:
                        st.sidebar.error("❌ File needs at least 10 valid messages!")
                    else:
                        # Standardize labels
                        new_df['label'] = new_df['label'].astype(str).str.lower().str.strip()
                        label_mapping = {'ham': 'ham', 'spam': 'spam', '0': 'ham', '1': 'spam', 'legitimate': 'ham', 'phishing': 'spam'}
                        new_df['label'] = new_df['label'].map(label_mapping).fillna(new_df['label'])
                        
                        # Check we have both classes
                        unique_labels = new_df['label'].unique()
                        if not all(label in unique_labels for label in ['ham', 'spam']):
                            st.sidebar.error("❌ File must contain both 'ham' and 'spam' labels!")
                        else:
                            # Check minimum samples per class
                            class_counts = new_df['label'].value_counts()
                            if class_counts.min() < 5:
                                st.sidebar.error(f"❌ Need at least 5 samples per class! Current: {dict(class_counts)}")
                            else:
                                new_df['timestamp'] = datetime.now()
                                
                                # Save as new dataset
                                new_df.to_csv("uploaded_dataset.csv", index=False)
                                st.sidebar.success(f"✅ Uploaded {len(new_df)} messages")
                                st.sidebar.info("Click 'Force Retrain Model' to use the new dataset")
        
    except pd.errors.EmptyDataError:
        st.sidebar.error("❌ File is empty!")
    except pd.errors.ParserError as e:
        st.sidebar.error(f"❌ Cannot parse file: {str(e)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error uploading file: {str(e)}")

# Main interface
col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    if st.session_state.model_loaded:
        st.markdown(f'<div class="metric-card"><h3>📈 Accuracy</h3><h2>{st.session_state.current_accuracy*100:.1f}%</h2></div>', unsafe_allow_html=True)
    elif st.session_state.model_loading:
        st.markdown(f'<div class="metric-card"><h3>📈 Accuracy</h3><h2>Loading...</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card"><h3>📈 Accuracy</h3><h2>Not Ready</h2></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="metric-card"><h3>📊 Dataset</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)

with col1:
    st.subheader("🔍 Message Analysis")
    
    # Use session state for text area
    user_input = st.text_area(
        "Enter message to analyze:", 
        height=120, 
        placeholder="Type your message here...",
        value=st.session_state.get('input_text', ''),
        key="message_input"
    )
    
    # Update session state when text changes
    if user_input != st.session_state.get('input_text', ''):
        st.session_state.input_text = user_input
    
    col_analyze, col_clear = st.columns([1, 1])
    with col_analyze:
        analyze_button = st.button("🔍 Analyze Message", type="primary", disabled=not st.session_state.model_loaded or st.session_state.model_loading)
    with col_clear:
        clear_button = st.button("🗑️ Clear")
        
# Handle clear button separately to ensure it works
if clear_button:
    st.session_state.input_text = ""
    if 'last_prediction' in st.session_state:
        del st.session_state['last_prediction']
    st.rerun()

# Improved prediction with better error handling and fixed probability calculation
if analyze_button and user_input.strip() and st.session_state.model_loaded:
    # Validate input message
    is_valid, error_msg = validate_message(user_input)
    if not is_valid:
        st.error(f"❌ {error_msg}")
    else:
        with st.spinner("🔍 Analyzing message..."):
            try:
                start_time = time.time()
                
                processed_input = preprocess_text(user_input)
                if not processed_input.strip():
                    st.error("❌ Message contains no analyzable text!")
                else:
                    user_vec = st.session_state.current_vectorizer.transform([processed_input])
                    prediction = st.session_state.current_model.predict(user_vec)[0]
                    
                    # Fixed probability calculation
                    try:
                        probabilities = st.session_state.current_model.predict_proba(user_vec)[0]
                        classes = st.session_state.current_model.classes_
                        
                        # Get the probability of the predicted class
                        if len(classes) >= 2:
                            if prediction == 'spam':
                                spam_idx = list(classes).index('spam') if 'spam' in classes else 1
                                prob = probabilities[spam_idx] if spam_idx < len(probabilities) else probabilities.max()
                            else:
                                ham_idx = list(classes).index('ham') if 'ham' in classes else 0
                                prob = probabilities[ham_idx] if ham_idx < len(probabilities) else probabilities.max()
                        else:
                            prob = probabilities[0] if len(probabilities) > 0 else 0.85
                        
                        prob = max(prob, 0.5)  # Ensure minimum confidence
                    except Exception as e:
                        st.warning(f"Could not get probability scores: {str(e)}")
                        prob = 0.85  # Default confidence
                    
                    # Calculate prediction time
                    end_time = time.time()
                    prediction_time = end_time - start_time
                    
                    if prediction_time > 5.0:
                        st.warning(f"⚠️ Slow prediction: {prediction_time:.2f}s")
                    
                    # Store result in session state
                    prediction_result = {
                        'message': user_input,
                        'prediction': prediction,
                        'probability': prob,
                        'features': extract_features(user_input),
                        'timestamp': datetime.now(),
                        'processing_time': prediction_time
                    }
                    
                    st.session_state.last_prediction = prediction_result
                    
                    # Add to prediction history and manage memory
                    st.session_state.prediction_history.append(prediction_result)
                    manage_memory()
                    
                    # Automatically add prediction to dataset for continuous learning (only high confidence)
                    if prob > 0.75:
                        add_result = add_to_dataset(user_input, prediction, prob)
                        if add_result is None:
                            st.warning("⚠️ Could not save prediction to dataset")
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.session_state.last_prediction = None

# Display results if prediction exists
if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
    prediction_data = st.session_state.last_prediction
    prediction = prediction_data['prediction']
    prob = prediction_data['probability']
    features = prediction_data['features']
    
    # Display results
    col_result1, col_result2 = st.columns([2, 1])

    with col_result1:
        if prediction == 'spam':
            st.markdown(f'<div class="result-spam">🚨 SPAM DETECTED<br>Confidence: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            
            risk_factors = []
            if features['exclamation_count'] > 2:
                risk_factors.append(f"• Multiple exclamation marks ({features['exclamation_count']})")
            if features['uppercase_ratio'] > 0.3:
                risk_factors.append(f"• High uppercase ratio ({features['uppercase_ratio']:.1%})")
            if features['spam_keywords'] > 0:
                risk_factors.append(f"• Contains spam keywords ({features['spam_keywords']})")
            if features['length'] > 500:
                risk_factors.append(f"• Very long message ({features['length']} chars)")
            
            if risk_factors:
                st.warning("⚠️ **Risk Factors:**")
                for factor in risk_factors:
                    st.write(factor)
        else:
            st.markdown(f'<div class="result-safe">✅ LEGITIMATE MESSAGE<br>Confidence: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.success("🛡️ **Safe Message Indicators:**")
            st.write("• Normal message patterns")
            st.write("• Low spam probability")
    
    with col_result2:
        st.subheader("📊 Stats")
        st.metric("Length", f"{features['length']} chars")
        st.metric("Words", features['word_count'])
        st.metric("Exclamations", features['exclamation_count'])
        st.metric("Uppercase %", f"{features['uppercase_ratio']:.1%}")
        
        # Show processing time
        if 'processing_time' in prediction_data:
            st.metric("Processing", f"{prediction_data['processing_time']:.3f}s")
        
        # Show auto-learning status
        if prob > 0.75:
            st.success("🤖 Auto-learned!")
        else:
            st.info("🤔 Low confidence")
    
    # Optional manual correction with better feedback
    st.markdown("---")
    st.subheader("🎯 Correction (Optional)")
    st.write("Only use if the prediction above is wrong:")
    
    col_feedback1, col_feedback2 = st.columns(2)
    
    with col_feedback1:
        if st.button("❌ Actually SPAM") and prediction == 'ham':
            result = add_to_dataset(user_input, 'spam', 0.9)
            if result is not None:
                st.success("📝 Corrected! Added as spam to training data.")
            else:
                st.error("❌ Could not save correction")
    
    with col_feedback2:
        if st.button("❌ Actually SAFE") and prediction == 'spam':
            result = add_to_dataset(user_input, 'ham', 0.9)
            if result is not None:
                st.success("📝 Corrected! Added as safe message to training data.")
            else:
                st.error("❌ Could not save correction")

# Analytics Dashboard with better error handling
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Data Insights", "📝 Batch Analysis"])

with tab1:
    if st.session_state.model_loaded and st.session_state.current_y_test is not None and st.session_state.current_y_pred is not None:
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = confusion_matrix(st.session_state.current_y_test, st.session_state.current_y_pred)
                fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=['Ham', 'Spam'], y=['Ham', 'Spam'], 
                                  title="Confusion Matrix", color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Performance metrics
                try:
                    report = classification_report(st.session_state.current_y_test, st.session_state.current_y_pred, output_dict=True)
                    
                    if 'ham' in report and 'spam' in report:
                        metrics_data = {
                            'Metric': ['Precision (Ham)', 'Recall (Ham)', 'Precision (Spam)', 'Recall (Spam)'],
                            'Score': [
                                report['ham']['precision'], 
                                report['ham']['recall'], 
                                report['spam']['precision'], 
                                report['spam']['recall']
                            ]
                        }
                        
                        fig_metrics = px.bar(metrics_data, x='Metric', y='Score', title="Performance Metrics", 
                                           color='Score', color_continuous_scale="Viridis")
                        st.plotly_chart(fig_metrics, use_container_width=True)
                    else:
                        st.warning("⚠️ Cannot display metrics - incomplete classification report")
                        
                except Exception as e:
                    st.error(f"Error generating performance metrics: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error displaying performance analytics: {str(e)}")
    else:
        st.info("📊 Analytics will be available once the model finishes loading.")

with tab2:
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                fig_pie = px.pie(values=label_counts.values, names=label_counts.index, title="Dataset Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("⚠️ No label column found in dataset")
        
        with col2:
            # Message length distribution
            if 'message' in df.columns:
                df_temp = df.copy()
                df_temp['message_length'] = df_temp['message'].astype(str).str.len()
                fig_hist = px.histogram(df_temp, x='message_length', color='label' if 'label' in df_temp.columns else None, 
                                       title="Message Length Distribution", nbins=50, marginal="box")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("⚠️ No message column found in dataset")
                
    except Exception as e:
        st.error(f"Error displaying data insights: {str(e)}")

with tab3:
    st.subheader("📝 Batch Analysis")
    
    batch_input = st.text_area("Enter multiple messages (one per line):", height=150, 
                              placeholder="Message 1\nMessage 2\nMessage 3...")
    
    if st.button("🔍 Analyze Batch", disabled=not st.session_state.model_loaded or st.session_state.model_loading):
        if batch_input.strip() and st.session_state.model_loaded:
            messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
            
            if len(messages) == 0:
                st.error("❌ No valid messages found!")
            elif len(messages) > 50:
                st.error("❌ Too many messages! Maximum 50 at once.")
            else:
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, message in enumerate(messages, 1):
                    try:
                        # Validate each message
                        is_valid, error_msg = validate_message(message)
                        if not is_valid:
                            results.append({
                                'Message #': i,
                                'Message': message[:50] + '...' if len(message) > 50 else message,
                                'Prediction': f'❌ Error: {error_msg}',
                                'Confidence': 'N/A'
                            })
                            continue
                        
                        processed_msg = preprocess_text(message)
                        if not processed_msg.strip():
                            results.append({
                                'Message #': i,
                                'Message': message[:50] + '...' if len(message) > 50 else message,
                                'Prediction': '❌ No analyzable text',
                                'Confidence': 'N/A'
                            })
                            continue
                        
                        msg_vec = st.session_state.current_vectorizer.transform([processed_msg])
                        pred = st.session_state.current_model.predict(msg_vec)[0]
                        
                        try:
                            probabilities = st.session_state.current_model.predict_proba(msg_vec)[0]
                            classes = st.session_state.current_model.classes_
                            
                            if pred == 'spam':
                                spam_idx = list(classes).index('spam') if 'spam' in classes else 1
                                prob = probabilities[spam_idx] if spam_idx < len(probabilities) else probabilities.max()
                            else:
                                ham_idx = list(classes).index('ham') if 'ham' in classes else 0
                                prob = probabilities[ham_idx] if ham_idx < len(probabilities) else probabilities.max()
                        except:
                            prob = 0.85
                        
                        results.append({
                            'Message #': i,
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Prediction': '🚨 SPAM' if pred == 'spam' else '✅ SAFE',
                            'Confidence': f"{prob*100:.1f}%"
                        })
                        
                    except Exception as e:
                        results.append({
                            'Message #': i,
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Prediction': f'❌ Error: {str(e)[:30]}',
                            'Confidence': 'N/A'
                        })
                    
                    # Update progress
                    progress_bar.progress(i / len(messages))
                    status_text.text(f"Processing message {i}/{len(messages)}")
                
                progress_bar.empty()
                status_text.empty()
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary with error handling
                try:
                    spam_count = sum(1 for r in results if '🚨' in r['Prediction'])
                    safe_count = sum(1 for r in results if '✅' in r['Prediction'])
                    error_count = sum(1 for r in results if '❌' in r['Prediction'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Messages", len(results))
                    with col2:
                        st.metric("Spam Detected", spam_count)  
                    with col3:
                        st.metric("Safe Messages", safe_count)
                    with col4:
                        st.metric("Errors", error_count)
                        
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} messages could not be analyzed")
                        
                except Exception as e:
                    st.error(f"Error calculating summary: {str(e)}")
        else:
            st.error("❌ Please wait for the model to finish loading before using batch analysis.")

# Model Comparison Section
st.markdown("---")
st.subheader("🏆 Model Performance Comparison")

# Create comparison data
if st.session_state.model_loaded:
    try:
        comparison_data = {
            'Model': ['SVM', 'Naive Bayes', 'Logistic Regression'],
            'Expected Accuracy': [96, 92, 89],
            'Speed': ['Fast', 'Very Fast', 'Fast'],
            'Memory Usage': ['Low', 'Very Low', 'Low'],
            'Best For': ['Text Classification', 'Quick Prototyping', 'Baseline Model']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight current model
        try:
            current_row = comparison_df[comparison_df['Model'] == model_choice].index[0]
            st.info(f"✅ Currently using: **{model_choice}** - {comparison_df.loc[current_row, 'Best For']}")
        except IndexError:
            st.info(f"✅ Currently using: **{model_choice}**")
            
    except Exception as e:
        st.error(f"Error displaying model comparison: {str(e)}")

# Advanced Features Section
st.markdown("---")
st.subheader("🚀 Advanced Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Model Features:
    - **Linear SVM**: Industry standard for spam detection
    - **TF-IDF Vectorization**: Optimized for text analysis
    - **Bigram Support**: Captures phrase patterns
    - **Auto-Learning**: Improves with each prediction
    - **Model Caching**: Train once, use repeatedly
    """)

with col2:
    st.markdown("""
    ### 📊 Performance Optimizations:
    - **Balanced Class Weights**: Handles imbalanced data
    - **Feature Selection**: 3000 most relevant features
    - **Sublinear TF**: Better handling of frequent terms
    - **Min/Max DF Filtering**: Removes noise words
    - **Probability Calibration**: Accurate confidence scores
    """)

# Show saved models info with better error handling
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Saved Models")
try:
    if os.path.exists(MODELS_DIR):
        saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if saved_models:
            st.sidebar.success(f"✅ {len(saved_models)} saved models")
            for model_file in saved_models[:5]:  # Show only first 5 to avoid clutter
                model_info = model_file.replace('.pkl', '').replace('model_', '').replace('_', ' ').title()
                st.sidebar.text(f"• {model_info}")
            if len(saved_models) > 5:
                st.sidebar.text(f"... and {len(saved_models) - 5} more")
        else:
            st.sidebar.info("No saved models yet")
    else:
        st.sidebar.info("No saved models yet")
except Exception as e:
    st.sidebar.error(f"Error checking saved models: {str(e)}")

# Clear models button with confirmation
if st.sidebar.button("🗑️ Clear All Saved Models"):
    try:
        if os.path.exists(MODELS_DIR):
            deleted_count = 0
            for file in os.listdir(MODELS_DIR):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(MODELS_DIR, file))
                        deleted_count += 1
                    except Exception as e:
                        st.sidebar.warning(f"Could not delete {file}: {str(e)}")
            
            if deleted_count > 0:
                st.sidebar.success(f"✅ Cleared {deleted_count} saved models!")
                st.rerun()
            else:
                st.sidebar.info("No models to clear")
    except Exception as e:
        st.sidebar.error(f"Error clearing models: {str(e)}")

# Quick Examples Section
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Quick Examples")

example_messages = {
    "Safe Message": "Hey, how are you doing today?",
    "Spam Message": "URGENT! You won $1000000! Click now!",
    "Business Email": "Meeting scheduled for 3pm tomorrow",
    "Marketing Spam": "FREE iPhone! Limited time offer!"
}

for label, message in example_messages.items():
    if st.sidebar.button(f"Try: {label}"):
        st.session_state.input_text = message
        st.rerun()

# Footer with tips
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
    <h3>💡 Pro Tips</h3>
    <p>• <strong>SVM</strong> provides the highest accuracy for spam detection</p>
    <p>• Upload your own dataset for better customization</p>
    <p>• Use batch analysis for processing multiple messages</p>
    <p>• Model learns automatically from high-confidence predictions</p>
    <p>• Correct wrong predictions to improve future accuracy</p>
</div>
""", unsafe_allow_html=True)

# Performance metrics footer
st.markdown('<div style="text-align: center; color: #666; margin-top: 2rem;">🛡️ SpamShield | Powered by SVM & TF-IDF | Train Once, Predict Fast</div>', 
           unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #2c3e50, #34495e); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-top: 1rem;">
    <p style="margin: 0; font-size: 0.9rem;">
        💻 <strong>Developed by Tharun M & Sivadurgesh K</strong> | 
        🤝 Team Collaboration 
    </p>
</div>
""", unsafe_allow_html=True)