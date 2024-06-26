## "Enter password to unlock your login keyring The login keyring did not get unlocked when you logged into your computer."
chrome, slack等のアプリケーションでは通常はLogin keyringsがdefaultで使用される. Login keyringはlogin時のpasswordによって解除される. しかしなんらかの理由により, login keyringがdeafultになっていない場合, アプリケーション起動時にdefault keyringsを勝手に生成して, これを解除しようとするためこのようなエラーが発生する.

解決策は, windows key + search (password) default keyringsを削除して, login keyringをdefaultにすることである.

https://itsfoss.com/ubuntu-keyring/
