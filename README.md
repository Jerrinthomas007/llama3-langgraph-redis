# 🧠 llama3-redis-qdrant-chat

A **production-ready, scalable AI chatbot** that uses:

- ⚡ [Groq](https://console.groq.com/) + **LLaMA3** for blazing-fast LLM inference
- 🧠 **Redis** for short-term chat memory
- 🌐 Custom DuckDuckGo HTML search and smart scraping (no third-party DuckDuckGo lib)
- 📚 **LangGraph** for stateful, multi-step chat workflows
- 💻 React + Vite frontend for a clean chat UI

---

## ⚙️ Features

- 🔁 Real-time chat history using Redis
- 🌍 Online search using custom HTML scraping from DuckDuckGo
- 📄 Webpage content extraction using `trafilatura`
- 🌦️ Live weather support via DNS Toys
- 🧠 LLM-powered reasoning to decide between online search or direct response
- ✨ Extensible LangGraph architecture for multi-step chains
- 💬 Responsive frontend running on Vite (port `5173`)

---

## 🧠 How Online Search Works

This chatbot **does not rely on any DuckDuckGo Python library**.

It uses:

- **BeautifulSoup** to parse DuckDuckGo's HTML results (`https://html.duckduckgo.com/html/?q=...`)
- **Trafilatura** to extract clean text from webpages
- A separate LLM step to select the most relevant link for scraping

---

## 📁 Project Structure

```
.
├── src/
│   ├── backend/
│   │   ├── chatbot.py           # LangGraph state machine
│   │   ├── tools/
│   │   │   ├── online_search.py # Custom DuckDuckGo + scraping
│   │   │   └── dnstoys.py       # Weather info via DNS
│   │   ├── core/                # Redis setup
│   │   └── main.py              # FastAPI entry point (optional)
│   └── shared/                  # Typed request/response schemas
├── frontend/                    # Vite + React frontend
├── .env.sample                  # Sample env variables
```

---

## 🚀 Getting Started

### 🐍 Backend Setup

```bash
# Step 1: Create and activate Python environment
conda create -n llama3-redis-qdrant-chat python=3.10 -y
conda activate llama3-redis-qdrant-chat

# Step 2: Install backend dependencies
pip install -r requirements.txt

# Step 3: Set up environment variables
cp .env.sample .env
# 👉 Edit the new .env file and add your Groq API key:
# GROQ_API_KEY=your_groq_api_key_here

# Step 4: Start the backend
uvicorn src.backend.main:app --reload
```

> Ensure Redis is running locally on port `6379`

---

### 💻 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

> The frontend will be available at: `http://localhost:5173`

---

## 🧠 LangGraph Workflow

1. 🔄 Load user history from Redis
2. 🧠 Decide via LLaMA3 if external search is needed
3. 🌍 If needed, search via custom DuckDuckGo scraper and pick best result
4. 🕸 Scrape that result using `trafilatura`
5. 🤖 Generate response using prompt + history + context
6. 💾 Store updated history in Redis

---


## 🔧 Environment Setup

Edit `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
```

---


## 🤝 Contributing

Got improvements or ideas? PRs and issues are welcome!

---

## 📜 License

MIT © [Jerrin Thomas](https://github.com/yourusername)