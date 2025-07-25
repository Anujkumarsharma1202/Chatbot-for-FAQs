# 🤖 FAQ Chatbot using NLP

This project is part of the **Artificial Intelligence Internship at CodeAlpha**.  
The goal is to build a chatbot that can answer frequently asked questions using Natural Language Processing techniques.

---

## 🧠 Project Description

The FAQ Chatbot uses:
- **NLTK** for text preprocessing (lowercasing, stopword removal)
- **TF-IDF Vectorization** for feature extraction
- **Cosine Similarity** to match user queries with predefined FAQs
- **Flask** to serve a simple web-based chat interface

If a user's input closely matches a question in the FAQ database, the corresponding answer is displayed. Otherwise, the bot informs the user that it doesn't understand.

---

## 💡 Features

- Predefined set of FAQs (loaded from `faqs.json`)
- NLP-based text preprocessing
- Real-time matching using cosine similarity
- Simple and clean chat UI with Flask
- Fully offline: no external API required
- Compatible with **Python 3.10+**

---

## 📂 Project Structure

