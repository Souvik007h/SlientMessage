# 🕵️‍♂️ Silent Message

🔗 **Live Demo**: [https://silentmessage-app.streamlit.app/](https://silentmessage-app.streamlit.app/)

**Silent Message** is a steganography-based web application that enables secure and seamless embedding of secret text messages inside images without altering their visual appearance. Built with Python, OpenCV, and Streamlit, it ensures real-time performance — no external storage required!

---

## 🧠 Features

- 🔐 **Steganography-based security**: Hide secret messages in plain sight by embedding them in image pixels.
- 🖼️ **Image integrity**: The modified (stego) image looks visually identical to the original.
- ⚡ **Real-time processing**: Everything happens in memory — no files are saved on disk.
- 🧰 **Built with**:  
  - `Python` for backend logic  
  - `OpenCV` for pixel-level image processing  
  - `Streamlit` for a simple and elegant web interface

---

## 🚀 How It Works

### 🔏 Encoding (Hiding the Message)
1. Upload a cover image.
2. Enter your secret message.
3. Click “Encode” — the app embeds your message into the image.

### 🔓 Decoding (Retrieving the Message)
1. Upload the stego image.
2. Click “Decode” — the app extracts and displays the hidden message.

---

## 📸 Screenshot

![silentmessage-app streamlit app](https://github.com/user-attachments/assets/55297fdb-c89c-49d5-a0fb-6a24004e96f3)


---

## 🛠️ Installation

To run locally:

```bash
git clone https://github.com/your-username/silent-message.git
cd silent-message
pip install -r requirements.txt
streamlit run app.py
```
---
## 🧩 Use Cases
- Confidential communication
- Sending secure notes or passwords
- Educational demos on steganography
---
## 🛡️ Disclaimer
This project is for educational and informational purposes only. Do not use it for illegal or unethical activities.
---
## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
