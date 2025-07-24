#  Neural Intrusion Detection System (NIDS) + Threat Intelligence

An AI-powered Intrusion Detection System that goes beyond log scanning — it learns behavior, detects anomalies in real-time, and enriches alerts with threat intelligence using advanced neural models and large language models (LLMs).

---

##  Features

- **Log Anomaly Detection** using pretrained LSTM/Transformer models (e.g., LogAnomaly, BERT)
- **Network Traffic Analysis** using autoencoders to detect abnormal patterns in PCAP/log data
- **Threat Intelligence Summarization** with OpenAI GPT/BART models for real-time threat enrichment
- **VirusTotal API Integration** for live threat validation
- **Real-Time Visualization** through a custom Streamlit dashboard

---

##  Tech Stack

- **Language**: Python
- **Models**: LSTM, Transformers (HuggingFace), Autoencoders
- **Libraries**: PyTorch, Keras, HuggingFace Transformers, Streamlit, Requests
- **APIs**: VirusTotal, OpenAI / Cohere (LLMs for summarization)
- **Deployment**: Local or Streamlit Cloud

---

##  Project Structure

```bash
nids-project/
├── models/
│   ├── log_anomaly_model.py
│   ├── autoencoder.py
├── data/
│   ├── sample_logs.log
│   ├── pcap_samples.pcap
├── utils/
│   ├── parser.py
│   ├── anomaly_scoring.py
├── dashboard/
│   └── app.py             # Streamlit app
├── threat_intel/
│   └── summarizer.py      # GPT/BART threat insights
├── README.md
├── requirements.txt
