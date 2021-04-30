import requests


class TelegramBotNotifier:
    """
    Send notification using Telegram Bot API.

    Telegram bot can be created with @BotFather (https://t.me/botfather).
    """
    def __init__(self, token: str, chat: str) -> None:
        print("Configuring Telegram Bot Notifier...")
        self.entry_point = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat = chat

    def notify(self, subject: str, text: str) -> None:
        print("Sending Telegram Bot notification...")
        print("Message:", '"""', text, '"""', sep="\n")

        data = {
            "text": f"<b>{subject}</b>\n\n{text}",
            "chat_id": self.chat,
            "parse_mode": "html",
            "disable_web_page_preview": True
        }

        r = requests.post(self.entry_point, json=data)
        if r.status_code != 200:
            raise Exception(r.status_code, r.text)
        
        print("Sent!", r.text)

class MultiNotifier:
    def __init__(self, notifiers=None):
        if notifiers is not None:
            self.notifiers = notifiers
        else:
            self.notifiers = []

    def add_notifier(self, notifier):
        self.notifiers.append(notifier)

    def notify(self, subject: str, text: str) -> None:
        print("Triggering all notification methods...")
        problems = []
        nsuccess, nfail = 0, 0
        for notifier in self.notifiers:
            try:
                notifier.notify(subject, text)
                nsuccess += 1
            except Exception as e:
                problems.append((notifier, e))
                nfail += 1
        print(f"{nsuccess} notification methods triggered, {nfail} failed.")
        if problems != []:
            raise Exception("Some notification methods failed.", *problems)

NOTIFIER = MultiNotifier()
TELEGRAM_ACCESS_TOKEN = "737239960:AAEUK0C2Vb211gPRUWMffgrv1j6IBC0FuUw" 
TELEGRAM_DESTINATION  = 552384750 

NOTIFIER.add_notifier(TelegramBotNotifier(
   token=TELEGRAM_ACCESS_TOKEN,
   chat=TELEGRAM_DESTINATION))