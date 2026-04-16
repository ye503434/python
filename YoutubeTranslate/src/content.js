let lastText = "" ;
const gasURL = "https://script.google.com/macros/s/AKfycbzfnDAeTX--db2wNhQ8hwK0GU3ZZwKPXKvq0i8izwpYnzCNLNXmh-VoxOuSy_coLiqb/exec";
console.log("【系統訊息】YouTube 翻譯外掛已成功載入！時間：", new Date().toLocaleTimeString());
//抓取Youtube cc字幕
function getSubtitleOnScreen(){
    const segments = document.querySelectorAll('.ytp-caption-segment')
    if (segments.length === 0 ) return "";
    let fullText = "";
    segments.forEach(s =>{ 
        if(s && s.innerText){
            fullText += s.innerText + " ";
        }
    });
    return fullText.trim(); //.trim是去頭尾空格
}

//google翻譯 URL連接
async function translateText(text){
    try{
        const url = `${gasURL}?text=${encodeURIComponent(text)}&source=ko&target=zh-TW`;
        const response = await fetch(url);
        const result = await response.text()
        return result;
    }catch(e){
        console.error("翻譯出錯:",e)
        return("翻譯失敗")
    }
}

//顯示在螢幕上
function showSubtitleOnScreen(text) {
    let subtitleDiv = document.querySelector('.my-custom-subtitle');
    
    if (!subtitleDiv) {
        subtitleDiv = document.createElement('div');
        subtitleDiv.className = 'my-custom-subtitle';  
        document.body.appendChild(subtitleDiv);
    }
    
    subtitleDiv.innerText = text;
    subtitleDiv.style.display = "block";
}

//韓翻中 主邏輯
setInterval(async () => {
    try {
        const result = await chrome.storage.local.get(['translateEnabled']);
        const isEnabled = result.translateEnabled !== false;
        const subtitleDiv = document.querySelector('.my-custom-subtitle');

        if (!isEnabled) {
            if (subtitleDiv) subtitleDiv.style.display = "none";
            return;
        }

        const currentText = getSubtitleOnScreen();

        if (currentText === "") {
            if (subtitleDiv) subtitleDiv.style.display = "none";
            return; 
        }

        if (currentText !== lastText) {
            const tempText = currentText; 
            lastText = currentText;       
            
            console.log("偵測到新字幕:", tempText);

            if (tempText.includes('。') || tempText.includes('.') || 
                tempText.includes('?') || tempText.length >= 5) {
                
                console.log("正在翻譯...");
                const translated = await translateText(tempText);
                console.log(translated);
                showSubtitleOnScreen(translated);
            }
        }
    } catch (e) {
        console.error("執行發生錯誤:", e);
    }
}, 1000);
