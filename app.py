from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import os

app = Flask(__name__)

# ดึงค่าจาก Environment Variables (เพื่อความปลอดภัย)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler('3ff9ebb61e1dabaced81d4ab6cb05f76') # Channel Secret ของคุณ

from transformers import CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer

model_name = "Kanyasiri/wangchanberta-wongnai-3class"

# Load specifically as Camembert
config = CamembertConfig.from_pretrained(model_name)
model = CamembertForSequenceClassification.from_pretrained(model_name)
tokenizer = CamembertTokenizer.from_pretrained(model_name)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    
    # วิเคราะห์ความรู้สึก
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # แมพ Label: 0=บวก, 1=ปกติ, 2=ลบ
    labels = {0: "เชิงบวก 😊", 1: "ทั่วไป 😐", 2: "เชิงลบ 😡"}
    result = labels.get(prediction, "ไม่ทราบผล")
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"ผลวิเคราะห์รีวิว: {result}")
    )

if __name__ == "__main__":

    app.run()
