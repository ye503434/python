document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('toggleService');

    //打開選單時，先確認目前狀態
    chrome.storage.local.get(['translateEnabled'],(result) =>{
        toggle.checked = result.translateEnabled !== false;
    });

    //切換開關時，儲存
    toggle.addEventListener('change', ()=>{
        chrome.storage.local.set({ translateEnabled:toggle.checked});
    });
});