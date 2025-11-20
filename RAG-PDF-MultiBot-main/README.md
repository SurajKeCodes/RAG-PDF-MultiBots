# 📚 Multi-PDF Chatbot + Multi-Bot Chat

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.30+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **A powerful AI-powered application that combines PDF document analysis with specialized chatbots using Google's Gemini models**

<img width="1919" height="942" alt="image" src="https://github.com/user-attachments/assets/add63ff7-dad6-4f71-baff-f16567bb912a" />

 <img width="1918" height="1023" alt="image" src="https://github.com/user-attachments/assets/0f5df9ef-fdc3-4bee-9dd1-3e3a335fc596" />


## 🌟 Overview

This application provides two main functionalities:

1. **PDF Chat (RAG)**: Upload and chat with your PDF documents using Retrieval-Augmented Generation
2. **Specialized Bots**: Interact with purpose-built AI assistants optimized for different tasks

## ✨ Key Features

### 📄 PDF Chat System

- **Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **RAG Implementation**: Advanced Retrieval-Augmented Generation using FAISS vector store
- **Intelligent Chunking**: Smart text splitting with overlap for better context preservation
- **Contextual Answers**: Get precise answers based only on your uploaded documents

### 🧠 Specialized AI Bots

- **💻 Code Bot**: Expert programming assistant for coding questions and debugging
- **🤖 General Bot**: Knowledgeable assistant for general queries and conversations
- **📚 Study Bot**: Patient academic tutor for learning and educational support
- **💝 Emotional Bot**: Empathetic companion for emotional support and well-being

### 🎯 Advanced Features

- **Conversation History**: Persistent chat history for each bot with context awareness
- **Clean UI**: Modern, responsive interface with tabbed navigation
- **Real-time Processing**: Fast PDF indexing and instant responses
- **Session Management**: Automatic session handling and conversation persistence

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **AI Models**: Google Gemini (1.5-flash, 1.5-flash-latest, 1.5-flash-8b-latest)
- **Vector Database**: FAISS
- **Text Processing**: LangChain, RecursiveCharacterTextSplitter
- **PDF Processing**: PyPDF2
- **Embeddings**: Google Generative AI Embeddings

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.8 or higher
- Google API key for Gemini models
- Minimum 4GB RAM (8GB+ recommended for better performance)

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Surajkecode/Multi-PDF-Chatbot-Bots.git
   cd Multi-PDF-Chatbot-Bots
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Create required directories**
   ```bash
   mkdir img
   ```

## 📖 Usage

1. **Start the application**

   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

3. **Using PDF Chat**

   - Upload PDF files using the sidebar
   - Click "Submit & Process" to index the documents
   - Ask questions about your PDFs in the "PDF Chat" tab

4. **Using Specialized Bots**
   - Switch to the "Bots" tab
   - Select your desired bot from the dropdown
   - Start chatting with your chosen AI assistant

## 📁 Project Structure

```
Multi-PDF-Chatbot-Bots/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # Project documentation
├── img/                  # Image assets          # Footer image
└── faiss_index/         # Vector store (auto-generated)
```

## 🔧 Configuration

### Bot Models Configuration

The application uses different Gemini models for specialized tasks:

```python
BOT_CATALOG = {
    "💻 Code Bot"        # Programming assistance
    "🤖 General Bot"     # General conversations
    "📚 Study Bot"       # Educational support
    "💝 Emotional Bot"   # Emotional support
}
```

### System Prompts

Each bot has optimized system prompts for their specific roles:

- **Code Bot**: Provides production-ready code with detailed explanations
- **General Bot**: Delivers structured, informative responses
- **Study Bot**: Breaks down complex concepts with examples
- **Emotional Bot**: Offers empathetic, supportive interactions

## 📸 Screenshots

<img width="1897" height="1010" alt="image" src="https://github.com/user-attachments/assets/236928bf-d8de-4d8f-a710-da89eb3990e0" />

<img width="1338" height="570" alt="image" src="https://github.com/user-attachments/assets/ad725dee-a584-4303-8922-d02178b410c1" />

<img width="1919" height="1005" alt="image" src="https://github.com/user-attachments/assets/6c194775-fb3d-4500-9707-598331c757e5" />

<img width="1909" height="924" alt="image" src="https://github.com/user-attachments/assets/c8d4d6dd-795d-41db-b7ca-f4e9871460e8" />

<img width="1918" height="1011" alt="image" src="https://github.com/user-attachments/assets/03ec2852-7c88-4077-b66c-57d4c969d55e" />

<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/e60c166b-8c28-42e0-acfd-42f0b83f420b" />





## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed

   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Error**: Ensure your Google API key is properly set in the `.env` file

3. **PDF Processing Issues**: Check if your PDF files are not password-protected

4. **Memory Issues**: For large PDFs, consider reducing chunk size in the configuration

## 🔮 Future Enhancements

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Advanced conversation export features
- [ ] Custom bot creation interface
- [ ] Multi-language support
- [ ] Voice interaction capabilities
- [ ] Integration with more LLM providers

## 🌐 Connect with Me

📧 **Email**: [surajborkute.tech@gmail.com](mailto:surajborkute.tech@gmail.com)

💼 **LinkedIn**: [Suraj Borkute](https://www.linkedin.com/in/suraj-borkute-87597b270)

💻 **GitHub**: [Surajkecode](https://github.com/Surajkecode)

📱 **WhatsApp**: [Message Now](https://wa.me/919518772281) or 📞 +91 9518772281

---

⭐ **If you found this project helpful, please give it a star!**

_Built with ❤️ by Suraj Borkute_
