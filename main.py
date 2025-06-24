import pandas as pd

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels: 0 - fake, 1 - true
fake_df["label"] = 0
true_df["label"] = 1

# Combine both datasets
data = pd.concat([fake_df, true_df], axis=0)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

print("Dataset shape:", data.shape)
print("\nSample data:")
print(data.head())

# -----------------------
# Text preprocessing
# -----------------------
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Create a combined content column
data["content"] = data["title"] + " " + data["text"]
data["content_clean"] = data["content"].apply(clean_text)

print("\nCleaned text sample:")
print(data["content_clean"].head())

# -----------------------
# WordCloud Visualization
# -----------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Split real and fake news
real_news = data[data["label"] == 1]
fake_news = data[data["label"] == 0]

# Join text for WordClouds
real_text = " ".join(real_news["content_clean"])
fake_text = " ".join(fake_news["content_clean"])

# Generate WordClouds
real_wc = WordCloud(width=800, height=400, background_color="white").generate(real_text)
fake_wc = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(fake_text)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(real_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Real News WordCloud", fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(fake_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Fake News WordCloud", fontsize=16)

plt.tight_layout()
plt.show()

# -----------------------
# Traditional ML Models
# -----------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(data["content_clean"]).toarray()
y = data["label"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# Naive Bayes
nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)

print("\n📘 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\n📔 Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -----------------------
# Deep Learning Model
# -----------------------
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# Build a simple neural network
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate performance
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📘 Deep Learning Model Results: \nAccuracy: {accuracy:.4f}")

# -----------------------
# Training Visualization
# -----------------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("visuals/deep_learning_training_plot.png")
plt.show()
