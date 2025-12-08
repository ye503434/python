from fastapi import FastAPI
from fastapi.responses import Response

import coin.yfinace as yf

app = FastAPI()


# 在Terminal 輸入 uvicorn apiTest.main:app --reload
@app.get("/finance")
async def root(coin, start):
    png_data = yf.Sol.finance(coin, start)
    return Response(content=png_data, media_type="image/png")
