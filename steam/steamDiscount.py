import os

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import  MIMEMultipart

# 本地端執行需要 ， git Action上不需要
# from dotenv import load_dotenv
# load_dotenv()
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

def get_steam_deals():

    url = "https://store.steampowered.com/api/featuredcategories/?l=zh-tw"
    try:
        response = requests.get(url)
        data = response.json()

        #取得 specials類別中的遊戲
        specials = data.get('specials',{}).get('items',[])
        deals = []

        for item in specials:
            discount = item.get('discount_percent', 0)
            #折扣超過50% 才回傳
            if discount >= 50:
                deal_info = {
                    'name': item.get('name'),
                    'discount': discount,
                    'original_price': item.get('original_price') / 100 ,
                    'final_price' : item.get('final_price') / 100,
                    'link': f"https://store.steampowered.com/app/{item.get('id')}"
                }
                deals.append(deal_info)
            return deals
    except Exception as e :
        print(f'抓取失敗: {e}')
        return[]

def send_email(deals):
    if not deals:
        print("沒有超過50%的特價遊戲")
        return

    #郵件內容(HTML格式)
    subject = "今日steam熱門遊戲 折扣50%以上"
    html_content = "<h2>以下是熱門特價遊戲：</h2><table border='1' style='border-collapse: collapse;'>"
    html_content += "<tr><th>遊戲名稱</th><th>折扣</th><th>原價</th><th>特價</th><th>連結</th></tr>"

    for deal in deals:
        html_content += f"""
        <tr>
            <td style='padding: 8px;'>{deal['name']}</td>
            <td style='padding: 8px; color: red;'>-{deal['discount']}%</td>
            <td style='padding: 8px;'>NT$ {deal['original_price']}</td>
            <td style='padding: 8px;'><b>NT$ {deal['final_price']}</b></td>
            <td style='padding: 8px;'><a href='{deal['link']}'>商店頁面</a></td>
        </tr>
    """
    html_content += "</table>"

    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html'))

    try:
        #設定 SMTP 伺服器
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(GMAIL_USER,GMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("發送成功")
    except Exception as e :
        print(f'發送失敗: {e}')

if __name__ == '__main__':
    sale_items = get_steam_deals()
    send_email(sale_items)










